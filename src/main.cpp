// to enable vulkan validation layers
// adb shell setprop debug.oculus.loadandinjectpackagedvvl.com.kadhem.vryume 1
// adb shell setprop debug.vvl.forcelayerlog 1
#include <cstdint>
#include <iterator>
#include <malloc.h>
#include <vulkan/vulkan_core.h>
#define XR_USE_PLATFORM_ANDROID
#define XR_USE_GRAPHICS_API_VULKAN
#include <android/log.h>
#include <jni.h>
#include <vulkan/vulkan.h>
#include <openxr/openxr.h> // needs to be after vulkan and android stuff
#include <openxr/openxr_platform.h>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <atomic>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <errno.h>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "vryume", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "vryume", __VA_ARGS__)

#define CHECK_VK(x)                                                            \
  if ((x) != VK_SUCCESS) {                                                     \
    LOGE("VK ERROR %d at %s:%d\n", x, __FILE__, __LINE__);                     \
    assert(false);                                                             \
  }

#define CHECK_XR(x)                                                            \
  if ((x) != XR_SUCCESS) {                                                     \
    LOGE("XR ERROR %d at %s:%d\n", x, __FILE__, __LINE__);                     \
    assert(false);                                                             \
  }

struct Swapchain {
  XrSwapchain xr_swapchain;
  XrSwapchainImageVulkan2KHR *images;
  uint32_t image_count;
  VkDescriptorSet *descriptor_sets; // one per image
  VkImageView *image_views;         // one per image
};

static XrInstance instance;
static XrSystemId systemId;
static VkInstance vkInstance;
static VkPhysicalDevice vkPhysicalDevice;
static uint32_t vkQueueFamilyIndex;
static VkDevice vkDevice;
static VkQueue vkQueue;
static VkCommandPool command_pool;
static XrSession session;
static uint32_t viewCount;
static XrViewConfigurationView *viewConfigurations;
static Swapchain *color_swapchains;
static Swapchain *depth_swapchains;
static VkFence *fence_exec; // one per view
static VkCommandBuffer *command_buffers;
static XrCompositionLayerDepthInfoKHR *composition_layer_depth_info;
static XrCompositionLayerProjectionView *composition_layer_projection_views;
static XrView *views;
static XrSpace xr_space_stage;
static VkPipelineLayout pipelineLayout;
static VkPipeline computePipeline;

static bool is_xr_session_running = false;
static bool should_run_framecycle = false;
uint32_t tick() {

  XrEventDataBuffer xr_event_buffer{XR_TYPE_EVENT_DATA_BUFFER};
  XrResult result = xrPollEvent(instance, &xr_event_buffer);
  while (result == XR_SUCCESS) {
    switch (xr_event_buffer.type) {

    case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
      XrEventDataSessionStateChanged *pxr_session_state_changed =
          reinterpret_cast<XrEventDataSessionStateChanged *>(&xr_event_buffer);
      if (pxr_session_state_changed->session != session) {
        LOGE("[XrProgram] Received session state changed for unknown "
             "session?!");
        break;
      }

      switch (pxr_session_state_changed->state) {
      case XR_SESSION_STATE_IDLE:
      case XR_SESSION_STATE_UNKNOWN: {
        LOGI("OpenXr state: UNKNOWN");
        should_run_framecycle = false;
        break;
      }

      case XR_SESSION_STATE_FOCUSED:
      case XR_SESSION_STATE_SYNCHRONIZED:
      case XR_SESSION_STATE_VISIBLE: {
        LOGI("OpenXr state: FOCUSED | SYNCHRONIZED | VISIBLE");
        should_run_framecycle = true;
        break;
      }

      case XR_SESSION_STATE_READY: {
        LOGI("OpenXr state: READY");
        if (!is_xr_session_running) {
          LOGI("Begining Xr Session");
          XrSessionBeginInfo xr_session_begin_info = {
              .type = XR_TYPE_SESSION_BEGIN_INFO,
              .primaryViewConfigurationType =
                  XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO};
          CHECK_XR(xrBeginSession(session, &xr_session_begin_info));
          is_xr_session_running = true;
        }

        should_run_framecycle = true;
        break;
      }

      case XR_SESSION_STATE_STOPPING: {
        LOGI("OpenXr state: STOPPING");
        if (is_xr_session_running) {
          LOGI("calling xrEndSession");
          CHECK_XR(xrEndSession(session));
          is_xr_session_running = false;
        }

        should_run_framecycle = false;
        break;
      }

      case XR_SESSION_STATE_LOSS_PENDING:
      case XR_SESSION_STATE_EXITING: {
        assert(!is_xr_session_running);
        LOGI("calling xrDestroySession");
        CHECK_XR(xrDestroySession(session));

        should_run_framecycle = false;
        break;
      }

      default: {
        LOGE("[XrProgram] SESSION_STATE_CHANGED: Unhandled event: %i",
             pxr_session_state_changed->state);
        break;
      }
      }

      break;
    }

    case XR_TYPE_EVENT_DATA_EVENTS_LOST: {
      auto *p_events_lost =
          reinterpret_cast<XrEventDataEventsLost *>(&xr_event_buffer);
      LOGE("[XrProgram] EVENTS_LOST: Lost events: %i",
           p_events_lost->lostEventCount);
      break;
    }

    case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING: {
      auto *_pxr_instance_loss_pending =
          reinterpret_cast<XrEventDataInstanceLossPending *>(&xr_event_buffer);

      is_xr_session_running = false;
      should_run_framecycle = false;
      break;
    }

    default: {
      LOGE("Unhandled event type: %u", xr_event_buffer.type);
      break;
    }
    }

    result = xrPollEvent(instance, &xr_event_buffer);
  }

  if (!should_run_framecycle) {
    usleep(100); // hack to not spin while session stopped
    return 0;
  }

  XrFrameState frame_state{XR_TYPE_FRAME_STATE};
  {
    XrFrameWaitInfo frame_wait_info = {
        .type = XR_TYPE_FRAME_WAIT_INFO,
    };
    CHECK_XR(xrWaitFrame(session, &frame_wait_info, &frame_state));
  }

  if (!frame_state.shouldRender) {
    LOGI("frame_state.shouldRender is false");
    return 0;
  }

  // should we ? assert(_out_view_count == viewCount);

  XrFrameBeginInfo frame_begin_info = {
      .type = XR_TYPE_FRAME_BEGIN_INFO,
  };
  CHECK_XR(xrBeginFrame(session, &frame_begin_info));

  CHECK_VK(
      vkWaitForFences(vkDevice, viewCount, fence_exec, VK_TRUE, UINT64_MAX));
  CHECK_VK(vkResetFences(vkDevice, viewCount, fence_exec));
  CHECK_VK(vkResetCommandPool(vkDevice, command_pool, 0));

  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = command_pool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = viewCount;

  CHECK_VK(vkAllocateCommandBuffers(vkDevice, &allocInfo, command_buffers));

  // views
  {
    XrViewState view_state = {
        .type = XR_TYPE_VIEW_STATE,
    };
    XrViewLocateInfo view_locate_info = {
        .type = XR_TYPE_VIEW_LOCATE_INFO,
        .viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
        .displayTime = frame_state.predictedDisplayTime,
        .space = xr_space_stage,
    };

    uint32_t _out_view_count;
    for (uint32_t i = 0; i < viewCount; i++) {
      views[i].type = XR_TYPE_VIEW;
    }

    CHECK_XR(xrLocateViews(session, &view_locate_info, &view_state, viewCount,
                           &_out_view_count, views));
  }

  for (uint32_t i = 0; i < viewCount; i++) {
    uint32_t aquired_color_image_index = 0;
    uint32_t aquired_depth_image_index = 0;
    XrSwapchainImageAcquireInfo swapchain_image_acquire_info = {
        .type = XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO,
    };
    // xrAcquireSwapchainImage call only gives us the index of the image
    // we will render to, for the purpose of recording commands. we still neeed
    // to call xrWaitSwapchainImage before submitting graphics commands that
    // write to the image.
    CHECK_XR(xrAcquireSwapchainImage(color_swapchains[i].xr_swapchain,
                                     &swapchain_image_acquire_info,
                                     &aquired_color_image_index));

    CHECK_XR(xrAcquireSwapchainImage(depth_swapchains[i].xr_swapchain,
                                     &swapchain_image_acquire_info,
                                     &aquired_depth_image_index));

    auto view_image_extent = VkExtent3D{
        .width = viewConfigurations[i].recommendedImageRectWidth,
        .height = viewConfigurations[i].recommendedImageRectHeight,
        .depth = 1,
    };

    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    CHECK_VK(vkBeginCommandBuffer(command_buffers[i], &beginInfo));

    vkCmdBindPipeline(command_buffers[i], VK_PIPELINE_BIND_POINT_COMPUTE,
                      computePipeline);

    VkDescriptorSet sets[2] = {
        color_swapchains[i].descriptor_sets[aquired_color_image_index],
        depth_swapchains[i].descriptor_sets[aquired_depth_image_index]};

    vkCmdBindDescriptorSets(command_buffers[i], VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout,
                            0, // firstSet
                            2, // setCount
                            sets, 0, nullptr);

    vkCmdDispatch(command_buffers[i],
                  viewConfigurations[i].recommendedImageRectWidth / 16,
                  viewConfigurations[i].recommendedImageRectHeight / 16, 1);

    CHECK_VK(vkEndCommandBuffer(command_buffers[i]));

    composition_layer_depth_info[i] = (XrCompositionLayerDepthInfoKHR){
        .type = XR_TYPE_COMPOSITION_LAYER_DEPTH_INFO_KHR,
        .subImage =
            {
                .swapchain = depth_swapchains[i].xr_swapchain,
                .imageRect = {.offset = {0, 0},
                              .extent = {(int32_t)view_image_extent.width,
                                         (int32_t)view_image_extent.height}},
            },
        .minDepth = 0.f,
        .maxDepth = 1.f,
        .nearZ = 0.01f,
        .farZ = 100.f,
    };
    composition_layer_projection_views[i] = {
        .type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW,
        .next = &composition_layer_depth_info[i],
        .pose = views[i].pose,
        .fov = views[i].fov,
        .subImage =
            {
                .swapchain = color_swapchains[i].xr_swapchain,
                .imageRect =
                    {
                        .offset = {0, 0},
                        .extent = {(int32_t)view_image_extent.width,
                                   (int32_t)view_image_extent.height},
                    },
            },
    };
  }

  for (uint32_t i = 0; i < viewCount; i++) {
    XrSwapchainImageWaitInfo color_swapchain_image_wait_info = {
        .type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO,
        .timeout = XR_INFINITE_DURATION,
    };
    CHECK_XR(xrWaitSwapchainImage(color_swapchains[i].xr_swapchain,
                                  &color_swapchain_image_wait_info));

    XrSwapchainImageWaitInfo depth_swapchain_image_wait_info = {
        .type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO,
        .timeout = XR_INFINITE_DURATION,
    };
    CHECK_XR(xrWaitSwapchainImage(depth_swapchains[i].xr_swapchain,
                                  &depth_swapchain_image_wait_info));
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffers[i],
    };
    CHECK_VK(vkQueueSubmit(vkQueue, 1, &submit_info, fence_exec[i]));

    XrSwapchainImageReleaseInfo color_swapchain_image_release_info = {
        .type = XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO,
    };
    CHECK_XR(xrReleaseSwapchainImage(color_swapchains[i].xr_swapchain,
                                     &color_swapchain_image_release_info));

    XrSwapchainImageReleaseInfo depth_swapchain_image_release_info = {
        .type = XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO,
    };
    CHECK_XR(xrReleaseSwapchainImage(depth_swapchains[i].xr_swapchain,
                                     &depth_swapchain_image_release_info));
  }

  XrCompositionLayerProjection composition_layer_projection = {
      .type = XR_TYPE_COMPOSITION_LAYER_PROJECTION,
      .space = xr_space_stage,
      .viewCount = viewCount,
      .views = composition_layer_projection_views,
  };

  XrCompositionLayerBaseHeader *layers_base[1] = {
      (XrCompositionLayerBaseHeader *)&composition_layer_projection};

  XrFrameEndInfo frame_end_info = {
      .type = XR_TYPE_FRAME_END_INFO,
      .displayTime = frame_state.predictedDisplayTime,
      .environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE,
      .layerCount = 1,
      .layers = layers_base};

  CHECK_XR(xrEndFrame(session, &frame_end_info));

  return 0;
}

extern "C" int engine_main(
    JavaVM *jvm, jobject main_activity,
    bool *android_requests_exit // android app received "destory" lifecycle //
                                // TODO, this should be an atomic of some sort
) {
  LOGI("Message from inside engine_main 7");

  LOGI("called with jvm=%p, main_activity=%p\n", (void *)jvm,
       (void *)main_activity);

  PFN_xrInitializeLoaderKHR xrInitializeLoaderKHR;
  xrGetInstanceProcAddr(XR_NULL_HANDLE, "xrInitializeLoaderKHR",
                        (PFN_xrVoidFunction *)&xrInitializeLoaderKHR);

  XrLoaderInitInfoAndroidKHR loaderInitInfo = {
      .type = XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR,
      .next = NULL,
      .applicationVM = jvm,
      .applicationContext = main_activity};
  xrInitializeLoaderKHR((const XrLoaderInitInfoBaseHeaderKHR *)&loaderInitInfo);

  // instance
  {
    XrInstanceCreateInfoAndroidKHR androidInfo = {
        .type = XR_TYPE_INSTANCE_CREATE_INFO_ANDROID_KHR,
        .applicationVM = jvm,
        .applicationActivity = main_activity,
    };
    const char *extensions[] = {
        XR_EXT_LOCAL_FLOOR_EXTENSION_NAME,
        XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME,
        XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME,
        XR_EXT_DEBUG_UTILS_EXTENSION_NAME,
    };
    XrInstanceCreateInfo instanceCreateInfo = {XR_TYPE_INSTANCE_CREATE_INFO};
    snprintf(instanceCreateInfo.applicationInfo.applicationName, 128, "VRYume");
    instanceCreateInfo.applicationInfo.applicationVersion = 1;
    snprintf(instanceCreateInfo.applicationInfo.engineName, 128, "NoEngine");
    instanceCreateInfo.applicationInfo.engineVersion = 1;
    instanceCreateInfo.applicationInfo.apiVersion = XR_API_VERSION_1_0;
    instanceCreateInfo.enabledExtensionCount = 4;
    instanceCreateInfo.enabledExtensionNames = extensions;
    instanceCreateInfo.next = &androidInfo;

    CHECK_XR(xrCreateInstance(&instanceCreateInfo, &instance));
  }

  // systemId
  XrSystemGetInfo systemInfo = {XR_TYPE_SYSTEM_GET_INFO};
  systemInfo.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
  CHECK_XR(xrGetSystem(instance, &systemInfo, &systemId) != XR_SUCCESS);

  {
    PFN_xrGetVulkanGraphicsRequirements2KHR
        xrGetVulkanGraphicsRequirements2KHR = NULL;
    xrGetInstanceProcAddr(
        instance, "xrGetVulkanGraphicsRequirements2KHR",
        (PFN_xrVoidFunction *)&xrGetVulkanGraphicsRequirements2KHR);

    XrGraphicsRequirementsVulkan2KHR graphicsRequirements = {
        .type = XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR};
    CHECK_XR(xrGetVulkanGraphicsRequirements2KHR(instance, systemId,
                                                 &graphicsRequirements));

    LOGI("XR: Min Vulkan API Version: %u.%u.%u\n",
         XR_VERSION_MAJOR(graphicsRequirements.minApiVersionSupported),
         XR_VERSION_MINOR(graphicsRequirements.minApiVersionSupported),
         XR_VERSION_PATCH(graphicsRequirements.minApiVersionSupported));
    LOGI("XR: Max Vulkan API Version: %u.%u.%u\n",
         XR_VERSION_MAJOR(graphicsRequirements.maxApiVersionSupported),
         XR_VERSION_MINOR(graphicsRequirements.maxApiVersionSupported),
         XR_VERSION_PATCH(graphicsRequirements.maxApiVersionSupported));
  }

  {
    const char *instExts[] = {
        "VK_KHR_surface",
        "VK_KHR_android_surface" // Needed on Quest/Android
    };

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "VRYume",
        .apiVersion = VK_API_VERSION_1_1,
    };

    VkInstanceCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
        .enabledExtensionCount = 2,
        .ppEnabledExtensionNames = instExts,
    };

    XrVulkanInstanceCreateInfoKHR xr_vulkan_instance_create_info = {
        .type = XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR,
        .systemId = systemId,
        .pfnGetInstanceProcAddr = &vkGetInstanceProcAddr,
        .vulkanCreateInfo = &ci,
        .vulkanAllocator = nullptr,
    };

    VkResult vk_result;
    PFN_xrCreateVulkanInstanceKHR xrCreateVulkanInstanceKHR;
    xrGetInstanceProcAddr(instance, "xrCreateVulkanInstanceKHR",
                          (PFN_xrVoidFunction *)&xrCreateVulkanInstanceKHR);
    CHECK_XR(xrCreateVulkanInstanceKHR(
        instance, &xr_vulkan_instance_create_info, &vkInstance, &vk_result));

    CHECK_VK(vk_result);
  }

  {
    PFN_xrGetVulkanGraphicsDevice2KHR xrGetVulkanGraphicsDevice2KHR;
    xrGetInstanceProcAddr(instance, "xrGetVulkanGraphicsDevice2KHR",
                          (PFN_xrVoidFunction *)&xrGetVulkanGraphicsDevice2KHR);

    XrVulkanGraphicsDeviceGetInfoKHR xr_vk_device_info = {
        .type = XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR,
        .systemId = systemId,
        .vulkanInstance = vkInstance,
    };
    CHECK_XR(xrGetVulkanGraphicsDevice2KHR(instance, &xr_vk_device_info,
                                           &vkPhysicalDevice) != XR_SUCCESS);
  }

  {
    uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice, &qcount, NULL);
    VkQueueFamilyProperties props[16];
    vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice, &qcount, props);

    vkQueueFamilyIndex = UINT32_MAX;

    for (uint32_t i = 0; i < qcount; i++) {
      VkQueueFlags f = props[i].queueFlags;
      if ((f & VK_QUEUE_GRAPHICS_BIT) && (f & VK_QUEUE_COMPUTE_BIT) &&
          (f & VK_QUEUE_TRANSFER_BIT)) {
        vkQueueFamilyIndex = i;
        break;
      }
    }

    if (vkQueueFamilyIndex == UINT32_MAX) {
      LOGE("No universal queue family found!\n");
      return -1;
    }
  }

  {
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = vkQueueFamilyIndex,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };

    const char *devExts[] = {"VK_KHR_swapchain"}; // required for XR swapchains

    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                              .queueCreateInfoCount = 1,
                              .pQueueCreateInfos = &qci,
                              .enabledExtensionCount = 1,
                              .ppEnabledExtensionNames = devExts};

    PFN_xrCreateVulkanDeviceKHR xrCreateVulkanDeviceKHR;
    xrGetInstanceProcAddr(instance, "xrCreateVulkanDeviceKHR",
                          (PFN_xrVoidFunction *)&xrCreateVulkanDeviceKHR);

    XrVulkanDeviceCreateInfoKHR xr_vulkan_device_create_info = {
        .type = XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR,
        .systemId = systemId,
        .pfnGetInstanceProcAddr = &vkGetInstanceProcAddr,
        .vulkanPhysicalDevice = vkPhysicalDevice,
        .vulkanCreateInfo = &dci,
        .vulkanAllocator = nullptr};

    VkResult vk_result;
    CHECK_XR(xrCreateVulkanDeviceKHR(instance, &xr_vulkan_device_create_info,
                                     &vkDevice, &vk_result));
    CHECK_VK(vk_result);

    vkGetDeviceQueue(vkDevice, vkQueueFamilyIndex, 0, &vkQueue);
  }

  {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags =
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // allows reuse/reset
    poolInfo.queueFamilyIndex = vkQueueFamilyIndex;

    CHECK_VK(vkCreateCommandPool(vkDevice, &poolInfo, nullptr, &command_pool));
  }

  {
    XrGraphicsBindingVulkan2KHR graphicsBinding = {
        .type = XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR};
    graphicsBinding.device = vkDevice;
    graphicsBinding.instance = vkInstance;
    graphicsBinding.physicalDevice = vkPhysicalDevice;
    graphicsBinding.queueFamilyIndex = vkQueueFamilyIndex;
    graphicsBinding.queueIndex = 0;

    XrSessionCreateInfo sessionCreateInfo = {
        .type = XR_TYPE_SESSION_CREATE_INFO,
        .createFlags = 0,
    };
    sessionCreateInfo.next = &graphicsBinding;
    sessionCreateInfo.systemId = systemId;

    CHECK_XR(xrCreateSession(instance, &sessionCreateInfo, &session));
  }

  {
    XrPosef xr_pose_identity = {
        .orientation =
            {
                .w = 1.f,
            },
    };

    // create reference mmap_reference_spaces
    XrReferenceSpaceCreateInfo xr_reference_space_create_info = {
        .type = XR_TYPE_REFERENCE_SPACE_CREATE_INFO,
        .referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE,
        .poseInReferenceSpace = xr_pose_identity,
    };
    CHECK_XR(xrCreateReferenceSpace(session, &xr_reference_space_create_info,
                                    &xr_space_stage));
  }

  {
    xrEnumerateViewConfigurationViews(instance, systemId,
                                      XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
                                      0, &viewCount, NULL);
    viewConfigurations = (XrViewConfigurationView *)malloc(
        sizeof(XrViewConfigurationView) * viewCount);

    for (uint32_t i = 0; i < viewCount; i++) {
      viewConfigurations[i].type = XR_TYPE_VIEW_CONFIGURATION_VIEW;
    }

    CHECK_XR(xrEnumerateViewConfigurationViews(
        instance, systemId, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
        viewCount, &viewCount, viewConfigurations));

    for (uint32_t i = 0; i < viewCount; i++) {
      LOGI("view[%d]: %d x %d", i,
           viewConfigurations[i].recommendedImageRectWidth,
           viewConfigurations[i].recommendedImageRectHeight);
    }
    LOGI("xrEnumerateViewConfigurationViews returned %u", viewCount);
  }

  color_swapchains = (Swapchain *)malloc(sizeof(Swapchain) * viewCount);
  for (uint32_t i = 0; i < viewCount; i++) {
    XrSwapchainCreateInfo swapchainInfo = {XR_TYPE_SWAPCHAIN_CREATE_INFO};
    swapchainInfo.usageFlags = XR_SWAPCHAIN_USAGE_UNORDERED_ACCESS_BIT |
                               XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT |
                               XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT;
    swapchainInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    swapchainInfo.sampleCount =
        viewConfigurations[i].recommendedSwapchainSampleCount;
    swapchainInfo.width = viewConfigurations[i].recommendedImageRectWidth;
    swapchainInfo.height = viewConfigurations[i].recommendedImageRectHeight;
    swapchainInfo.faceCount = 1;
    swapchainInfo.arraySize = 1;
    swapchainInfo.mipCount = 1;

    CHECK_XR(xrCreateSwapchain(session, &swapchainInfo,
                               &color_swapchains[i].xr_swapchain));

    CHECK_XR(xrEnumerateSwapchainImages(color_swapchains[i].xr_swapchain, 0,
                                        &color_swapchains[i].image_count,
                                        nullptr));

    color_swapchains[i].images = (XrSwapchainImageVulkan2KHR *)malloc(
        sizeof(XrSwapchainImageVulkan2KHR) * color_swapchains[i].image_count);
    color_swapchains[i].image_views = (VkImageView *)malloc(
        sizeof(VkImageView) * color_swapchains[i].image_count);
    color_swapchains[i].descriptor_sets = (VkDescriptorSet *)malloc(
        sizeof(VkDescriptorSet) * color_swapchains[i].image_count);

    for (uint32_t j = 0; j < color_swapchains[i].image_count; j++) {
      color_swapchains[i].images[j] = {.type =
                                           XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR};
    }

    uint32_t _swapchain_image_count;
    CHECK_XR(xrEnumerateSwapchainImages(
        color_swapchains[i].xr_swapchain, color_swapchains[i].image_count,
        &_swapchain_image_count,
        (XrSwapchainImageBaseHeader *)color_swapchains[i].images));

    for (uint32_t j = 0; j < color_swapchains[i].image_count; j++) {
      VkImageViewCreateInfo viewInfo = {
          .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
          .flags = 0,
          .image = color_swapchains[i].images[j].image,
          .viewType = VK_IMAGE_VIEW_TYPE_2D,
          .format = VK_FORMAT_R8G8B8A8_UNORM,
          .components =
              {
                  .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                  .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                  .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                  .a = VK_COMPONENT_SWIZZLE_IDENTITY,
              },
          .subresourceRange =
              {
                  .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                  .baseMipLevel = 0,
                  .levelCount = 1,
                  .baseArrayLayer = 0,
                  .layerCount = 1,
              },
      };

      CHECK_VK(vkCreateImageView(vkDevice, &viewInfo, nullptr,
                                 &color_swapchains[i].image_views[j]));
    }
  }

  depth_swapchains = (Swapchain *)malloc(sizeof(Swapchain) * viewCount);
  for (uint32_t i = 0; i < viewCount; i++) {
    XrSwapchainCreateInfo swapchain_create_info = {
        .type = XR_TYPE_SWAPCHAIN_CREATE_INFO,
        .usageFlags = XR_SWAPCHAIN_USAGE_SAMPLED_BIT |
                      XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                      XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT,
        .format =
            VK_FORMAT_D16_UNORM, // TODO: we are assuming this will be
                                 // supported, but the we should test that.
        .sampleCount = viewConfigurations[i].recommendedSwapchainSampleCount,
        .width = viewConfigurations[i].recommendedImageRectWidth,
        .height = viewConfigurations[i].recommendedImageRectHeight,
        .faceCount = 1,
        .arraySize = 1,
        .mipCount = 1,
    };
    CHECK_XR(xrCreateSwapchain(session, &swapchain_create_info,
                               &depth_swapchains[i].xr_swapchain));

    CHECK_XR(xrEnumerateSwapchainImages(depth_swapchains[i].xr_swapchain, 0,
                                        &depth_swapchains[i].image_count,
                                        nullptr));
    depth_swapchains[i].images = (XrSwapchainImageVulkan2KHR *)malloc(
        sizeof(XrSwapchainImageVulkan2KHR) * depth_swapchains[i].image_count);
    depth_swapchains[i].image_views = (VkImageView *)malloc(
        sizeof(VkImageView) * depth_swapchains[i].image_count);
    depth_swapchains[i].descriptor_sets = (VkDescriptorSet *)malloc(
        sizeof(VkDescriptorSet) * depth_swapchains[i].image_count);

    for (uint32_t j = 0; j < depth_swapchains[i].image_count; j++) {
      depth_swapchains[i].images[j] = {.type =
                                           XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR};
    }

    uint32_t _swapchain_image_count;
    CHECK_XR(xrEnumerateSwapchainImages(
        depth_swapchains[i].xr_swapchain, depth_swapchains[i].image_count,
        &_swapchain_image_count,
        (XrSwapchainImageBaseHeader *)depth_swapchains[i].images));

    for (uint32_t j = 0; j < depth_swapchains[i].image_count; j++) {
      VkImageViewCreateInfo viewInfo = {
          .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
          .flags = 0,
          .image = depth_swapchains[i].images[j].image,
          .viewType = VK_IMAGE_VIEW_TYPE_2D,
          .format = VK_FORMAT_D16_UNORM,
          .components =
              {
                  .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                  .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                  .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                  .a = VK_COMPONENT_SWIZZLE_IDENTITY,
              },

          .subresourceRange =
              {
                  .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                  .baseMipLevel = 0,
                  .levelCount = 1,
                  .baseArrayLayer = 0,
                  .layerCount = 1,
              },
      };

      CHECK_VK(vkCreateImageView(vkDevice, &viewInfo, nullptr,
                                 &depth_swapchains[i].image_views[j]));
    }
  }

  fence_exec = (VkFence *)malloc(sizeof(VkFence) * viewCount);
  for (uint32_t i = 0; i < viewCount; i++) {
    VkFenceCreateInfo fence_create_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    CHECK_VK(
        vkCreateFence(vkDevice, &fence_create_info, nullptr, &fence_exec[i]));
  }

  command_buffers =
      (VkCommandBuffer *)malloc(sizeof(VkCommandBuffer) * viewCount);
  composition_layer_depth_info = (XrCompositionLayerDepthInfoKHR *)malloc(
      sizeof(XrCompositionLayerDepthInfoKHR) * viewCount);
  composition_layer_projection_views =
      (XrCompositionLayerProjectionView *)malloc(
          sizeof(XrCompositionLayerProjectionView) * viewCount);
  views = (XrView *)malloc(sizeof(XrView) * viewCount);

  // pipelineLayout
  VkDescriptorSetLayout color_descriptor_set_layout;
  VkDescriptorSetLayout depth_descriptor_set_layout;
  {

    VkDescriptorSetLayoutBinding color_image_binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = NULL};

    VkDescriptorSetLayoutCreateInfo colorDescriptorSetLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &color_image_binding};

    CHECK_VK(vkCreateDescriptorSetLayout(vkDevice,
                                         &colorDescriptorSetLayoutInfo, nullptr,
                                         &color_descriptor_set_layout));

    VkDescriptorSetLayoutBinding depth_image_binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = NULL};

    VkDescriptorSetLayoutCreateInfo depthDescriptorSetLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &depth_image_binding};

    CHECK_VK(vkCreateDescriptorSetLayout(vkDevice,
                                         &depthDescriptorSetLayoutInfo, nullptr,
                                         &depth_descriptor_set_layout));

    VkDescriptorSetLayout set_layouts[] = {color_descriptor_set_layout,
                                           depth_descriptor_set_layout};

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2,
        .pSetLayouts = set_layouts,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr};

    CHECK_VK(vkCreatePipelineLayout(vkDevice, &pipelineLayoutInfo, NULL,
                                    &pipelineLayout));
  }

  // computePipeline
  {
    VkShaderModule shaderModule;
    {
      // TODO: this is not proper android, we should get the data folder path
      // from an api
      FILE *f = fopen(
          "/data/user/0/com.kadhem.vryume/files/assets/main.glsl.spv", "rb");
      if (!f) {
        LOGE("Error opening file: %s\n", strerror(errno));
        return 1;
      }

      fseek(f, 0, SEEK_END);
      size_t size = ftell(f);
      rewind(f);

      uint32_t *code = (uint32_t *)malloc(size);
      fread(code, 1, size, f);
      fclose(f);

      VkShaderModuleCreateInfo shaderModuleInfo = {
          .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
          .codeSize = size,
          .pCode = code};

      CHECK_VK(vkCreateShaderModule(vkDevice, &shaderModuleInfo, NULL,
                                    &shaderModule));

      free(code);
    }

    VkPipelineShaderStageCreateInfo stageInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shaderModule,
        .pName = "main"};

    VkComputePipelineCreateInfo pipelineInfo = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stageInfo,
        .layout = pipelineLayout};

    CHECK_VK(vkCreateComputePipelines(vkDevice, VK_NULL_HANDLE, 1,
                                      &pipelineInfo, NULL, &computePipeline));
  }

  // descriptor allocation and update
  {
    uint32_t total_color_swapchain_image_count = 0;
    uint32_t total_depth_swapchain_image_count = 0;

    for (int i = 0; i < viewCount; i++) {
      total_color_swapchain_image_count += color_swapchains[i].image_count;
      total_depth_swapchain_image_count += depth_swapchains[i].image_count;
    }

    uint32_t descriptor_count =
        total_color_swapchain_image_count + total_depth_swapchain_image_count;

    VkDescriptorPoolSize poolSize = {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                     .descriptorCount = descriptor_count};

    VkDescriptorPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = descriptor_count, // one descriptor per set
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize};

    VkDescriptorPool descriptorPool;
    CHECK_VK(
        vkCreateDescriptorPool(vkDevice, &poolInfo, NULL, &descriptorPool));

    for (uint32_t i = 0; i < viewCount; i++) {
      for (uint32_t j = 0; j < color_swapchains[i].image_count; j++) {
        VkDescriptorSetAllocateInfo allocInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &color_descriptor_set_layout};

        VkDescriptorSet descriptorSet;
        CHECK_VK(
            vkAllocateDescriptorSets(vkDevice, &allocInfo, &descriptorSet));

        VkDescriptorImageInfo imageInfo = {
            .sampler = VK_NULL_HANDLE,
            .imageView = color_swapchains[i].image_views[j],
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

        VkWriteDescriptorSet write = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptorSet,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &imageInfo};

        vkUpdateDescriptorSets(vkDevice, 1, &write, 0, nullptr);

        color_swapchains[i].descriptor_sets[j] = descriptorSet;
      }
    }
    for (uint32_t i = 0; i < viewCount; i++) {
      for (uint32_t j = 0; j < depth_swapchains[i].image_count; j++) {
        VkDescriptorSetAllocateInfo allocInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &depth_descriptor_set_layout};

        VkDescriptorSet descriptorSet;
        CHECK_VK(
            vkAllocateDescriptorSets(vkDevice, &allocInfo, &descriptorSet));

        VkDescriptorImageInfo imageInfo = {
            .sampler = VK_NULL_HANDLE,
            .imageView = depth_swapchains[i].image_views[j],
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL // we will write this from
                                                   // the shader direclty
        };

        VkWriteDescriptorSet write = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptorSet,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &imageInfo};

        vkUpdateDescriptorSets(vkDevice, 1, &write, 0, nullptr);
        depth_swapchains[i].descriptor_sets[j] = descriptorSet;
      }
    }
  }

  while (!*android_requests_exit) {
    tick();
  }

  // TODO: we should cleanup properly, android reuses the same process for
  // mulitple app invocation, we might have leftover state
  // don't forget to zero out the globals

  for (uint32_t i = 0; i < viewCount; i++) {
    xrDestroySwapchain(color_swapchains[i].xr_swapchain);
    xrDestroySwapchain(depth_swapchains[i].xr_swapchain);
  }
  free(viewConfigurations);
  free(color_swapchains);
  xrDestroySession(session);
  xrDestroyInstance(instance);

  LOGI("Cleaned up!\n");

  return 0;
}
