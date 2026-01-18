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
static VkImage *storageImage;
static VkDeviceMemory *storageMemory;
static VkFence fence_exec;
static VkCommandBuffer *command_buffers;
static XrCompositionLayerDepthInfoKHR *composition_layer_depth_info;
static XrCompositionLayerProjectionView *composition_layer_projection_views;
static XrView *views;
static XrSpace xr_space_stage;

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
    usleep(16); // hack to not spin while session stopped
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
  // should we ? assert(_out_view_count == viewCount);

  XrFrameBeginInfo frame_begin_info = {
      .type = XR_TYPE_FRAME_BEGIN_INFO,
  };
  CHECK_XR(xrBeginFrame(session, &frame_begin_info));

  CHECK_VK(vkWaitForFences(vkDevice, 1, &fence_exec, VK_TRUE, UINT64_MAX));
  CHECK_VK(vkResetFences(vkDevice, 1, &fence_exec));
  CHECK_VK(vkResetCommandPool(vkDevice, command_pool, 0));

  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = command_pool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = viewCount;

  CHECK_VK(vkAllocateCommandBuffers(vkDevice, &allocInfo, command_buffers));

  for (uint32_t i = 0; i < viewCount; i++) {
    uint32_t aquired_image_index = 0;
    XrSwapchainImageAcquireInfo swapchain_image_acquire_info = {
        .type = XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO,
    };
    // xrAcquireSwapchainImage call only gives us the index of the image
    // we will render to, for recording commands we still neeed to call
    // xrWaitSwapchainImage before submitting graphics commands that
    // write to the image.
    CHECK_XR(xrAcquireSwapchainImage(color_swapchains[i].xr_swapchain,
                                     &swapchain_image_acquire_info,
                                     &aquired_image_index));

    XrSwapchainImageWaitInfo swapchain_image_wait_info = {
        .type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO,
        .timeout = XR_INFINITE_DURATION,
    };
    CHECK_XR(xrWaitSwapchainImage(color_swapchains[i].xr_swapchain,
                                  &swapchain_image_wait_info));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VK(vkBeginCommandBuffer(command_buffers[i], &beginInfo));

    auto view_image_extent = VkExtent3D{
        .width = viewConfigurations[i].recommendedImageRectWidth,
        .height = viewConfigurations[i].recommendedImageRectHeight,
        .depth = 1,
    };

    {
      VkImageMemoryBarrier barriers[1] = {
          {
              .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
              .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
              .dstAccessMask =
                  VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
              .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .image = storageImage[i],
              .subresourceRange =
                  {
                      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                      .levelCount = 1,
                      .layerCount = 1,
                  },
          },
      };

      vkCmdPipelineBarrier(command_buffers[i],
                           VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr,
                           0, nullptr, 1, barriers);

      VkClearColorValue color = {
          .float32 = {1.0f, 1.0f, 0.0f, 1.0f}}; // RGBA = Red
      VkImageSubresourceRange range = {
          .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
          .baseMipLevel = 0,
          .levelCount = 1,
          .baseArrayLayer = 0,
          .layerCount = 1,
      };

      vkCmdClearColorImage(command_buffers[i], storageImage[i],
                           VK_IMAGE_LAYOUT_GENERAL, &color, 1, &range);
    }

    // TODO: check the access masks and stages, we are going to start
    // reusing the same storage image as last frame as soon as the
    // submited buffers trigger the fence
    VkImageMemoryBarrier barriers[2] = {
        {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = 0,
            .dstAccessMask =
                VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .image = storageImage[i],
            .subresourceRange =
                {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .levelCount = 1,
                    .layerCount = 1,
                },
        },
        {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = 0,
            .dstAccessMask =
                VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .image = color_swapchains[i].images[aquired_image_index].image,
            .subresourceRange =
                {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .levelCount = 1,
                    .layerCount = 1,
                },
        }};

    vkCmdPipelineBarrier(command_buffers[i], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 2, barriers);

    VkImageCopy region{};
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.layerCount = 1;
    region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.dstSubresource.layerCount = 1;
    region.extent = view_image_extent;

    vkCmdCopyImage(command_buffers[i], storageImage[i],
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   color_swapchains[i].images[aquired_image_index].image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // TODO: we should do the same thing about the depth

    CHECK_VK(vkEndCommandBuffer(command_buffers[i]));

    XrSwapchainImageReleaseInfo swapchain_image_release_info = {
        .type = XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO,
    };
    CHECK_XR(xrReleaseSwapchainImage(color_swapchains[i].xr_swapchain,
                                     &swapchain_image_release_info));

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
        //.next = &composition_layer_depth_info[i],
        .next = nullptr,
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

  VkSubmitInfo submit_info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = viewCount,
      .pCommandBuffers = command_buffers,
  };
  CHECK_VK(vkQueueSubmit(vkQueue, 1, &submit_info, fence_exec));

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

    LOGI("xrEnumerateViewConfigurationViews returned %u", viewCount);
  }

  color_swapchains = (Swapchain *)malloc(sizeof(Swapchain) * viewCount);
  for (uint32_t i = 0; i < viewCount; i++) {
    XrSwapchainCreateInfo swapchainInfo = {XR_TYPE_SWAPCHAIN_CREATE_INFO};
    swapchainInfo.usageFlags = XR_SWAPCHAIN_USAGE_SAMPLED_BIT |
                               XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT |
                               XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT;
    swapchainInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
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
    for (uint32_t j = 0; j < color_swapchains[i].image_count; j++) {
      color_swapchains[i].images[j] = {.type =
                                           XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR};
    }

    uint32_t _swapchain_image_count;
    CHECK_XR(xrEnumerateSwapchainImages(
        color_swapchains[i].xr_swapchain, color_swapchains[i].image_count,
        &_swapchain_image_count,
        (XrSwapchainImageBaseHeader *)color_swapchains[i].images));
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
    for (uint32_t j = 0; j < depth_swapchains[i].image_count; j++) {
      depth_swapchains[i].images[j] = {.type =
                                           XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR};
    }

    uint32_t _swapchain_image_count;
    CHECK_XR(xrEnumerateSwapchainImages(
        depth_swapchains[i].xr_swapchain, depth_swapchains[i].image_count,
        &_swapchain_image_count,
        (XrSwapchainImageBaseHeader *)depth_swapchains[i].images));
  }

  // create a storage image per view
  storageImage = (VkImage *)malloc(sizeof(VkImage) * viewCount);
  storageMemory = (VkDeviceMemory *)malloc(sizeof(VkDeviceMemory) * viewCount);
  for (uint32_t i = 0; i < viewCount; i++) {
    // Choose format you want for the storage image
    VkFormat imageFormat =
        VK_FORMAT_R8G8B8A8_SRGB; // TODO: this is a compatible format just to
                                 // test

    // Create the image
    auto extent = VkExtent3D{
        .width = viewConfigurations[i].recommendedImageRectWidth,
        .height = viewConfigurations[i].recommendedImageRectHeight,
        .depth = 1,
    };
    VkImageCreateInfo imgInfo{};
    imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.format = imageFormat;
    imgInfo.extent = extent;
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage = // VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                    // // TODO: we need storage and maybe sampled
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    CHECK_VK(vkCreateImage(vkDevice, &imgInfo, nullptr, &storageImage[i]));

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(vkDevice, storageImage[i], &memReq);

    // Find device-local memory type index
    uint32_t memoryTypeIndex = 0;
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(vkPhysicalDevice, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
      if ((memReq.memoryTypeBits & (1 << i)) &&
          (memProps.memoryTypes[i].propertyFlags &
           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
        memoryTypeIndex = i;
        break;
      }
    }

    VkMemoryAllocateInfo image_mem_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memReq.size,
        .memoryTypeIndex = memoryTypeIndex,
    };

    LOGI("image_mem_info: %p", &image_mem_info);

    CHECK_VK(vkAllocateMemory(vkDevice, &image_mem_info, nullptr,
                              &storageMemory[i]));
    CHECK_VK(vkBindImageMemory(vkDevice, storageImage[i], storageMemory[i], 0));
  }

  {
    VkFenceCreateInfo fence_create_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    CHECK_VK(vkCreateFence(vkDevice, &fence_create_info, nullptr, &fence_exec));
  }

  command_buffers =
      (VkCommandBuffer *)malloc(sizeof(VkCommandBuffer) * viewCount);
  composition_layer_depth_info = (XrCompositionLayerDepthInfoKHR *)malloc(
      sizeof(XrCompositionLayerDepthInfoKHR) * viewCount);
  composition_layer_projection_views =
      (XrCompositionLayerProjectionView *)malloc(
          sizeof(XrCompositionLayerProjectionView) * viewCount);
  views = (XrView *)malloc(sizeof(XrView) * viewCount);

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
