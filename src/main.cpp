// [Note: OpenXR camera configuration]
// In OpenXR, eye space origin is the eye position,
// +X is the right direction
// +Y is the up direction
// -Z is the forward direction
// the coordinate system is right handed.
// OpenXR defines the camera by offering each eye transfrom
// (translation/rotation) and 4 angles that define the view frustum.
//
// The angle absolute value is defined as the angle between the strait forward
// direction -Z in eye space and the sides of the frustum, so for example:
// fov.angleRight is the angle between -Z and the right side plane of the
// frustum.
//
// The sign is given by convenstion to help in computing the coordinate
// transfroms when computing the perspective matrix or other.
// Right, Up are positive, Left, Down are negative.

// to enable vulkan validation layers
// adb shell setprop debug.oculus.loadandinjectpackagedvvl.com.kadhem.vryume 1
// adb shell setprop debug.vvl.forcelayerlog 1
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <malloc.h>
#include <vulkan/vulkan_core.h>
#define XR_USE_GRAPHICS_API_VULKAN

#ifdef XR_USE_PLATFORM_ANDROID
#include <android/log.h>
#include <jni.h>
#endif

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
#include <signal.h>
#include <string.h>
#include <inttypes.h>
#include <cstdlib>
#include <cmath>
#include <renderdoc_app.h>
#include <dlfcn.h>

#ifdef XR_USE_PLATFORM_ANDROID
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "vryume", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "vryume", __VA_ARGS__)
#else
#define LOGI(...) printf(__VA_ARGS__)
#define LOGE(...) printf(__VA_ARGS__)
#endif

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
  uint32_t width;
  uint32_t height;
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
static VkFormat color_swapchain_format = VK_FORMAT_R8G8B8A8_UNORM;
static VkFormat depth_swapchain_format =
    VK_FORMAT_D16_UNORM; // TODO: this is likely broken we need to test properly
static PFN_vkCmdPipelineBarrier2KHR loaded_vkCmdPipelineBarrier2;
static uint32_t device_only_memory_type_index;
static VkMemoryPropertyFlags device_only_memory_type_property_flags;
static VkDescriptorPool rendering_descriptor_pool;
static VkDescriptorSet rendering_descriptor_set;
static VkSampler rendering_tile_sampler;
static RENDERDOC_API_1_4_1 *renderdoc_api = nullptr;

// TODO: this is shared with the shader, and as such we should define it
// somewhere global not duplciate it
constexpr float inter_sample_distance = 0.002; // in m
char *slow_concat(const char *a, const char *b) {
  size_t len_a = strlen(a);
  size_t len_b = strlen(b);

  char *result = (char *)malloc(len_a + len_b + 1); // +1 for '\0'
  assert(result);

  strcpy(result, a);
  strcat(result, b);

  return result;
}

VkShaderModule load_shader_module(const char *file_name) {
  VkShaderModule result;

#ifdef XR_USE_PLATFORM_ANDROID
  // TODO: this is not proper android, we should get the data folder path
  // from an api
  const char *base = "/data/user/0/com.kadhem.vryume/files/assets/";
#else
  const char *base = "./build/assets/";
#endif

  char *file_path = slow_concat(base, file_name);
  FILE *f = fopen(file_path, "rb");
  if (!f) {
    LOGE("Error opening file: %s\n", strerror(errno));
    std::exit(-1);
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

  CHECK_VK(vkCreateShaderModule(vkDevice, &shaderModuleInfo, NULL, &result));

  free(code);
  free(file_path);

  return result;
}

struct EvaluationPushConstant {
  float tile_pos[3]; // vec3 alignment=16
  float _pad; // pad to 16 so than next EvaluationPushConstant value in an array
              // starts at the correct alignment
};
struct EvaluationState {
  VkPipelineLayout pipeline_layout;
  VkPipeline pipeline;
  VkImageView tile_image_view;
  VkImage tile_image;
  VkDescriptorSet descriptor_set;
  VkDeviceMemory tile_image_memory;
  VkDescriptorSetLayout descriptor_set_layout;
  VkExtent3D tile_image_extent;
};

static EvaluationState evaluation_state;
// returns barrier needed to initialize the image before first shader access
VkImageMemoryBarrier2 initialize_evaluation_state() {
  {
    VkDescriptorSetLayoutBinding tile_image_binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = NULL};

    VkDescriptorSetLayoutCreateInfo evaluation_descriptor_set_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &tile_image_binding};

    CHECK_VK(vkCreateDescriptorSetLayout(
        vkDevice, &evaluation_descriptor_set_layout_info, nullptr,
        &evaluation_state.descriptor_set_layout));

    VkDescriptorSetLayout set_layouts[] = {
        evaluation_state.descriptor_set_layout};

    VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(EvaluationPushConstant),
    };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = set_layouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range};

    CHECK_VK(vkCreatePipelineLayout(vkDevice, &pipelineLayoutInfo, nullptr,
                                    &evaluation_state.pipeline_layout));
  }

  {

    VkShaderModule shader_module = load_shader_module("evaluation.glsl.spv");

    VkPipelineShaderStageCreateInfo stageInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_module,
        .pName = "main"};

    VkComputePipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stageInfo,
        .layout = evaluation_state.pipeline_layout};

    CHECK_VK(vkCreateComputePipelines(vkDevice, VK_NULL_HANDLE, 1,
                                      &pipeline_info, NULL,
                                      &evaluation_state.pipeline));
  }
  {
    evaluation_state.tile_image_extent = {
        .width = 1000, .height = 1000, .depth = 1000};

    VkImageCreateInfo image_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_3D,
        .format = VK_FORMAT_R8_UNORM,
        // TODO image size must be divisible by shader group size, we need a way
        // to make this less error prone
        .extent = evaluation_state.tile_image_extent,
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    CHECK_VK(vkCreateImage(vkDevice, &image_info, nullptr,
                           &evaluation_state.tile_image));
    VkMemoryRequirements memory_requirements;
    vkGetImageMemoryRequirements(vkDevice, evaluation_state.tile_image,
                                 &memory_requirements);
    assert(memory_requirements.memoryTypeBits &
           (1u << device_only_memory_type_index));
    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memory_requirements.size,
        .memoryTypeIndex = device_only_memory_type_index};

    CHECK_VK(vkAllocateMemory(vkDevice, &alloc_info, nullptr,
                              &evaluation_state.tile_image_memory));

    vkBindImageMemory(vkDevice, evaluation_state.tile_image,
                      evaluation_state.tile_image_memory, 0);
  }
  {
    VkImageViewCreateInfo image_view_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = evaluation_state.tile_image,
        .viewType = VK_IMAGE_VIEW_TYPE_3D,
        .format = VK_FORMAT_R8_UNORM,
        .components =
            VkComponentMapping{
                .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = VK_COMPONENT_SWIZZLE_IDENTITY,
            },
        .subresourceRange =
            VkImageSubresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            }

    };
    CHECK_VK(vkCreateImageView(vkDevice, &image_view_info, nullptr,
                               &evaluation_state.tile_image_view));
  }

  {
    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = rendering_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &evaluation_state.descriptor_set_layout,
    };

    CHECK_VK(vkAllocateDescriptorSets(vkDevice, &alloc_info,
                                      &evaluation_state.descriptor_set));

    VkDescriptorImageInfo image_info = {
        .sampler = VK_NULL_HANDLE,
        .imageView = evaluation_state.tile_image_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL, // the layout our image will be
                                                // at when used by shader
    };

    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = evaluation_state.descriptor_set,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &image_info};

    vkUpdateDescriptorSets(vkDevice, 1, &write, 0, nullptr);
  }
  VkImageMemoryBarrier2 barrier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,

      .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
      .srcAccessMask = VK_ACCESS_2_NONE,

      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask =
          VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,

      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,

      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,

      .image = evaluation_state.tile_image,
      .subresourceRange =
          {
              .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
              .baseMipLevel = 0,
              .levelCount = 1,
              .baseArrayLayer = 0,
              .layerCount = 1,
          },
  };
  return barrier;
}

struct RenderingPushConstant {
  float camera_orientation[4];
  float camera_pos[3];
  float tanLeft; // starts at 28
  float tanRight;
  float tanUp;
  float tanDown;
  float _pad;
  float tile_pos[3]; // starts at 48
  float _pad2;
  float tile_extent[3];
  float _pad3;
  float resolution[2]; // starts at 80
  float _pad4[2];      // to align struct to 16
};
static_assert(sizeof(RenderingPushConstant) == 96);
static_assert(sizeof(RenderingPushConstant) < 128, "PushConstant limit");

static bool is_xr_session_running = false;
static bool should_run_framecycle = false;
uint32_t tick() {

  XrEventDataBuffer xr_event_buffer{.type = XR_TYPE_EVENT_DATA_BUFFER};
  XrResult result = xrPollEvent(instance, &xr_event_buffer);
  while (result == XR_SUCCESS) {
    switch (xr_event_buffer.type) {

    case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
      XrEventDataSessionStateChanged *pxr_session_state_changed =
          reinterpret_cast<XrEventDataSessionStateChanged *>(&xr_event_buffer);
      if (pxr_session_state_changed->session != session) {
        LOGE("Unkown session\n");
        break;
      }

      switch (pxr_session_state_changed->state) {
      case XR_SESSION_STATE_IDLE:
      case XR_SESSION_STATE_UNKNOWN: {
        LOGI("OpenXr state: UNKNOWN\n");
        should_run_framecycle = false;
        break;
      }

      case XR_SESSION_STATE_FOCUSED:
      case XR_SESSION_STATE_SYNCHRONIZED:
      case XR_SESSION_STATE_VISIBLE: {
        LOGI("OpenXr state: FOCUSED | SYNCHRONIZED | VISIBLE\n");
        should_run_framecycle = true;
        break;
      }

      case XR_SESSION_STATE_READY: {
        LOGI("OpenXr state: READY\n");
        if (!is_xr_session_running) {
          LOGI("Begining Xr Session\n");
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
        LOGI("OpenXr state: STOPPING\n");
        if (is_xr_session_running) {
          LOGI("calling xrEndSession\n");
          CHECK_XR(xrEndSession(session));
          is_xr_session_running = false;
        }

        should_run_framecycle = false;
        break;
      }

      case XR_SESSION_STATE_LOSS_PENDING:
      case XR_SESSION_STATE_EXITING: {
        assert(!is_xr_session_running);
        LOGI("calling xrDestroySession\n");
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
      LOGE("[XrProgram] EVENTS_LOST: Lost events: %i\n",
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
      LOGE("Unhandled event type: %u\n", xr_event_buffer.type);
      break;
    }
    }

    xr_event_buffer = {.type = XR_TYPE_EVENT_DATA_BUFFER};
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

  // TODO
  if (!frame_state.shouldRender) {
    LOGI("frame_state.shouldRender is false\n");
    //   return 0;
  }

  // should we ? assert(_out_view_count == viewCount);

  if (renderdoc_api)
    renderdoc_api->StartFrameCapture(NULL, NULL);

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
      views[i] = {.type = XR_TYPE_VIEW};
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

    // vkCmdPipelineBarrier2
    {
      VkImageMemoryBarrier2 barriers[] = {
          {
              .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,

              .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
              .srcAccessMask = VK_ACCESS_2_NONE,

              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,

              .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,

              .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
              .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,

              .image =
                  color_swapchains[i].images[aquired_color_image_index].image,

              .subresourceRange =
                  {
                      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                      .baseMipLevel = 0,
                      .levelCount = 1,
                      .baseArrayLayer = 0,
                      .layerCount = 1,
                  },
          },
          {
              .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,

              .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
              .srcAccessMask = VK_ACCESS_2_NONE,

              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,

              .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,

              .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
              .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,

              .image =
                  depth_swapchains[i].images[aquired_depth_image_index].image,

              .subresourceRange =
                  {
                      .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                      .baseMipLevel = 0,
                      .levelCount = 1,
                      .baseArrayLayer = 0,
                      .layerCount = 1,
                  },
          }};

      VkDependencyInfo depInfo = {
          .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
          .imageMemoryBarrierCount = 2,
          .pImageMemoryBarriers = barriers,
      };

      loaded_vkCmdPipelineBarrier2(command_buffers[i], &depInfo);
    }

    vkCmdBindPipeline(command_buffers[i], VK_PIPELINE_BIND_POINT_COMPUTE,
                      computePipeline);

    VkDescriptorSet sets[3] = {
        color_swapchains[i].descriptor_sets[aquired_color_image_index],
        depth_swapchains[i].descriptor_sets[aquired_depth_image_index],
        rendering_descriptor_set,
    };

    vkCmdBindDescriptorSets(command_buffers[i], VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout,
                            0, // firstSet
                            3, // setCount
                            sets, 0, nullptr);

    auto orientation = views[i].pose.orientation;
    auto pos = views[i].pose.position;
    RenderingPushConstant rendering_push_constant = {
        //.camera_orientation = {orientation.x, orientation.y, orientation.z,
        //                       orientation.w},
        .camera_orientation = {0, 1, 0, 0.0000463},
        .camera_pos = {pos.x + 0.5f, pos.y + 0.5f,
                       pos.z + 0.5f}, // TODO, offset just for debugging
        .tanLeft = tanf(views[i].fov.angleLeft),
        .tanRight = tanf(views[i].fov.angleRight),
        .tanUp = tanf(views[i].fov.angleUp),
        .tanDown = tanf(views[i].fov.angleDown),
        .tile_pos = {0.0, 0.0, 0.0},
        .tile_extent =
            {
                (float)((evaluation_state.tile_image_extent.width - 1) *
                        inter_sample_distance),
                (float)((evaluation_state.tile_image_extent.height - 1) *
                        inter_sample_distance),
                (float)((evaluation_state.tile_image_extent.depth - 1) *
                        inter_sample_distance),
            },
        .resolution = {(float)color_swapchains[i].width,
                       (float)color_swapchains[i].height},

    };
    vkCmdPushConstants(command_buffers[i], pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, // offset
                       sizeof(RenderingPushConstant), &rendering_push_constant);

    vkCmdDispatch(command_buffers[i],
                  viewConfigurations[i].recommendedImageRectWidth / 16,
                  viewConfigurations[i].recommendedImageRectHeight / 16, 1);

    // vkCmdPipelineBarrier2
    {
      VkImageMemoryBarrier2 barriers[] = {
          {
              .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,

              .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,

              .dstStageMask = VK_PIPELINE_STAGE_2_NONE,
              .dstAccessMask = VK_ACCESS_2_NONE,

              .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
              .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,

              .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
              .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,

              .image =
                  color_swapchains[i].images[aquired_color_image_index].image,

              .subresourceRange =
                  {
                      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                      .baseMipLevel = 0,
                      .levelCount = 1,
                      .baseArrayLayer = 0,
                      .layerCount = 1,
                  },
          },
          {
              .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,

              .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,

              .dstStageMask = VK_PIPELINE_STAGE_2_NONE,
              .dstAccessMask = VK_ACCESS_2_NONE,

              .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
              .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,

              .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
              .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,

              .image =
                  depth_swapchains[i].images[aquired_depth_image_index].image,

              .subresourceRange =
                  {
                      .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                      .baseMipLevel = 0,
                      .levelCount = 1,
                      .baseArrayLayer = 0,
                      .layerCount = 1,
                  },
          }};

      VkDependencyInfo depInfo = {
          .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
          .imageMemoryBarrierCount = 2,
          .pImageMemoryBarriers = barriers,
      };

      loaded_vkCmdPipelineBarrier2(command_buffers[i], &depInfo);
    }

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

  if (renderdoc_api)
    renderdoc_api->EndFrameCapture(NULL, NULL);

  return 0;
}

XrBool32 XRAPI_PTR openxrDebugCallback(
    XrDebugUtilsMessageSeverityFlagsEXT severity,
    XrDebugUtilsMessageTypeFlagsEXT types,
    const XrDebugUtilsMessengerCallbackDataEXT *data, void *userData) {
  fprintf(stderr, "XR: %s\n\n", data->message);
  return XR_FALSE;
}

VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
              void *pUserData) {
  fprintf(stderr, "VK: id:%d, message:%s\n\n", pCallbackData->messageIdNumber,
          pCallbackData->pMessage);
  for (uint32_t i = 0; i < pCallbackData->objectCount; i++) {
    const VkDebugUtilsObjectNameInfoEXT *obj = &pCallbackData->pObjects[i];

    fprintf(stderr, "  Object[%u]: type=%d handle=0x%llx name=%s\n", i,
            obj->objectType, (unsigned long long)obj->objectHandle,
            obj->pObjectName ? obj->pObjectName : "NULL");
  }
  // if (pCallbackData->messageIdNumber == 328638116) {
  //   return VK_TRUE;
  // }
  //  Return VK_FALSE = do NOT abort the call
  return VK_FALSE;
}
extern "C" int engine_main(
#ifdef XR_USE_PLATFORM_ANDROID
    JavaVM *jvm, jobject main_activity,
    bool *android_requests_exit // android app received "destory" lifecycle,
                                // TODO, this should be an atomic of some sort
#endif
) {
  LOGI("Message from inside engine_main\n");

  // setup RenderDoc
  {
    // Look for the already-injected renderdoc library
    void *mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
    if (mod) {
      // Fetch the function pointer to get the API
      pRENDERDOC_GetAPI RENDERDOC_GetAPI =
          (pRENDERDOC_GetAPI)dlsym(mod, "RENDERDOC_GetAPI");

      // Populate the API structure (using version 1.4.1 as an example)
      int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_4_1,
                                 (void **)&renderdoc_api);
      if (ret == 1) {
        LOGI("RenderDoc API successfully hooked!\n");
      }
    } else {
      LOGI("RenderDoc not detected. Running normally.\n");
    }
  }

#ifdef XR_USE_PLATFORM_ANDROID

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
#endif

  // instance
  {
#ifdef XR_USE_PLATFORM_ANDROID
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
    // const char *layers[] = {"XR_APILAYER_LUNARG_core_validation"};

    XrInstanceCreateInfo instanceCreateInfo = {XR_TYPE_INSTANCE_CREATE_INFO};
    snprintf(instanceCreateInfo.applicationInfo.applicationName, 128, "VRYume");
    instanceCreateInfo.applicationInfo.applicationVersion = 1;
    snprintf(instanceCreateInfo.applicationInfo.engineName, 128, "NoEngine");
    instanceCreateInfo.applicationInfo.engineVersion = 1;
    instanceCreateInfo.applicationInfo.apiVersion = XR_API_VERSION_1_0;
    instanceCreateInfo.enabledExtensionCount = 4;
    instanceCreateInfo.enabledExtensionNames = extensions;
    // instanceCreateInfo.enabledApiLayerCount = 1;
    // instanceCreateInfo.enabledApiLayerNames = layers;
    instanceCreateInfo.next = &androidInfo;
#else
    const char *extensions[] = {
        XR_EXT_LOCAL_FLOOR_EXTENSION_NAME,
        XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME,
        XR_EXT_DEBUG_UTILS_EXTENSION_NAME,
    };
    const char *layers[] = {"XR_APILAYER_LUNARG_core_validation"};

    XrInstanceCreateInfo instanceCreateInfo = {XR_TYPE_INSTANCE_CREATE_INFO};
    snprintf(instanceCreateInfo.applicationInfo.applicationName, 128, "VRYume");
    instanceCreateInfo.applicationInfo.applicationVersion = 1;
    snprintf(instanceCreateInfo.applicationInfo.engineName, 128, "NoEngine");
    instanceCreateInfo.applicationInfo.engineVersion = 1;
    instanceCreateInfo.applicationInfo.apiVersion = XR_API_VERSION_1_0;
    instanceCreateInfo.enabledExtensionCount = 3;
    instanceCreateInfo.enabledExtensionNames = extensions;
    instanceCreateInfo.enabledApiLayerCount = 1;
    instanceCreateInfo.enabledApiLayerNames = layers;
#endif

    CHECK_XR(xrCreateInstance(&instanceCreateInfo, &instance));
  }

  {
    XrDebugUtilsMessengerCreateInfoEXT debugInfo{};
    debugInfo.type = XR_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;

    debugInfo.messageSeverities =
        XR_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        XR_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
        XR_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        XR_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

    debugInfo.messageTypes = XR_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             XR_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             XR_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
                             XR_DEBUG_UTILS_MESSAGE_TYPE_CONFORMANCE_BIT_EXT;

    debugInfo.userCallback = &openxrDebugCallback;
    debugInfo.userData = nullptr;

    PFN_xrCreateDebugUtilsMessengerEXT xrCreateDebugUtilsMessengerEXT = nullptr;
    xrGetInstanceProcAddr(
        instance, "xrCreateDebugUtilsMessengerEXT",
        (PFN_xrVoidFunction *)&xrCreateDebugUtilsMessengerEXT);

    XrDebugUtilsMessengerEXT messenger;
    xrCreateDebugUtilsMessengerEXT(instance, &debugInfo, &messenger);
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
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "VRYume",
        .apiVersion = VK_API_VERSION_1_2,
    };

#ifdef XR_USE_PLATFORM_ANDROID
    const char *instExts[] = {"VK_KHR_surface", "VK_KHR_android_surface"};

    VkInstanceCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
        .enabledExtensionCount = 2,
        .ppEnabledExtensionNames = instExts,
    };
#else
    const char *extensions[] = {"VK_KHR_surface",
                                VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
    const char *layers[] = {"VK_LAYER_KHRONOS_validation"};

    VkInstanceCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
        .enabledLayerCount = 1,
        .ppEnabledLayerNames = layers,
        .enabledExtensionCount = 2,
        .ppEnabledExtensionNames = extensions,
    };
#endif

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
    VkDebugUtilsMessengerCreateInfoEXT createInfo = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .flags = 0,
        .messageSeverity = // VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                           // VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,

        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,

        .pfnUserCallback = debugCallback,
    };

    PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT =
        (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            vkInstance, "vkCreateDebugUtilsMessengerEXT");

    VkDebugUtilsMessengerEXT debugMessenger;

    vkCreateDebugUtilsMessengerEXT(vkInstance, &createInfo, nullptr,
                                   &debugMessenger);
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

    VkPhysicalDeviceSynchronization2FeaturesKHR sync2_features = {
        .sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR,
        .pNext = NULL,
        .synchronization2 = VK_TRUE};

    const char *device_extensions[] = {"VK_KHR_swapchain",
                                       "VK_KHR_synchronization2"};

    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &sync2_features,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .enabledExtensionCount = 2,
        .ppEnabledExtensionNames = device_extensions,
    };

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

  // device_only_memory_type_index
  // device_only_memory_type_property_flags
  {
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(vkPhysicalDevice, &memory_properties);

    bool found_memory_type = false;
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
      VkMemoryPropertyFlags flags =
          memory_properties.memoryTypes[i].propertyFlags;
      if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
        device_only_memory_type_index = i;
        device_only_memory_type_property_flags = flags;
        found_memory_type = true;
        break;
      }
    }
    assert(found_memory_type);
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
      viewConfigurations[i] = {.type = XR_TYPE_VIEW_CONFIGURATION_VIEW};
    }

    CHECK_XR(xrEnumerateViewConfigurationViews(
        instance, systemId, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
        viewCount, &viewCount, viewConfigurations));

    for (uint32_t i = 0; i < viewCount; i++) {
      LOGI("view[%d]: %d x %d\n", i,
           viewConfigurations[i].recommendedImageRectWidth,
           viewConfigurations[i].recommendedImageRectHeight);
    }
    LOGI("xrEnumerateViewConfigurationViews returned %u\n", viewCount);
  }

  {
    loaded_vkCmdPipelineBarrier2 =
        (PFN_vkCmdPipelineBarrier2KHR)vkGetDeviceProcAddr(
            vkDevice, "vkCmdPipelineBarrier2KHR");
  }

  {

    uint32_t format_count = 0;

    CHECK_XR(xrEnumerateSwapchainFormats(session, 0, &format_count, NULL));

    int64_t *formats = (int64_t *)malloc(sizeof(int64_t) * format_count);

    CHECK_XR(xrEnumerateSwapchainFormats(session, format_count, &format_count,
                                         formats));

    printf("Supported OpenXR swapchain formats:\n");
    for (uint32_t i = 0; i < format_count; ++i) {
      printf("  VkFormat = %" PRId64 "\n", formats[i]);
    }

    free(formats);
  }

  color_swapchains = (Swapchain *)malloc(sizeof(Swapchain) * viewCount);
  for (uint32_t i = 0; i < viewCount; i++) {
    color_swapchains[i].width = viewConfigurations[i].recommendedImageRectWidth;
    color_swapchains[i].height =
        viewConfigurations[i].recommendedImageRectHeight;

    XrSwapchainCreateInfo swapchainInfo = {XR_TYPE_SWAPCHAIN_CREATE_INFO};
    swapchainInfo.usageFlags =
        XR_SWAPCHAIN_USAGE_UNORDERED_ACCESS_BIT |
        XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT; // added to get rid of some
                                                 // Vulkan validation layers
                                                 // errors complaining about
                                                 // missing image usage flags
    swapchainInfo.format = color_swapchain_format;

    swapchainInfo.sampleCount =
        viewConfigurations[i].recommendedSwapchainSampleCount;
    swapchainInfo.width = color_swapchains[i].width;
    swapchainInfo.height = color_swapchains[i].height;
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
          .format = color_swapchain_format,
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
    depth_swapchains[i].width = viewConfigurations[i].recommendedImageRectWidth;
    depth_swapchains[i].height =
        viewConfigurations[i].recommendedImageRectHeight;

    XrSwapchainCreateInfo swapchain_create_info = {
        .type = XR_TYPE_SWAPCHAIN_CREATE_INFO,
        .usageFlags = XR_SWAPCHAIN_USAGE_UNORDERED_ACCESS_BIT |
                      XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        .format = depth_swapchain_format,
        .sampleCount = viewConfigurations[i].recommendedSwapchainSampleCount,
        .width = depth_swapchains[i].width,
        .height = depth_swapchains[i].height,
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
          .format = depth_swapchain_format,
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
  VkDescriptorSetLayout rendering_set_layout;
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

    VkDescriptorSetLayoutBinding tile_descriptor_set_layout_binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    };

    VkDescriptorSetLayoutCreateInfo tile_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &tile_descriptor_set_layout_binding,
    };

    vkCreateDescriptorSetLayout(vkDevice, &tile_layout_info, nullptr,
                                &rendering_set_layout);

    VkDescriptorSetLayout set_layouts[] = {color_descriptor_set_layout,
                                           depth_descriptor_set_layout,
                                           rendering_set_layout};

    VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(RenderingPushConstant),
    };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 3,
        .pSetLayouts = set_layouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range};

    CHECK_VK(vkCreatePipelineLayout(vkDevice, &pipelineLayoutInfo, NULL,
                                    &pipelineLayout));
  }

  // computePipeline
  {
    VkShaderModule shaderModule = load_shader_module("main.glsl.spv");

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

  // rendering_descriptor_pool
  // a descriptor pool for things that are not present related
  {
    VkDescriptorPoolSize pool_storage_image_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 10, // TODO: need a way to size this properly
    };
    VkDescriptorPoolSize pool_combined_image_size = {
        .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 10, // TODO: need a way to size this properly
    };

    VkDescriptorPoolSize pool_sizes[2] = {pool_combined_image_size,
                                          pool_storage_image_size};
    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 10, // TODO: need a way to size this properly
        .poolSizeCount = 2,
        .pPoolSizes = pool_sizes};

    CHECK_VK(vkCreateDescriptorPool(vkDevice, &pool_info, nullptr,
                                    &rendering_descriptor_pool));
  }

  // initialize rendering related resources
  {
    CHECK_VK(vkResetCommandPool(vkDevice, command_pool, 0));
    VkCommandBuffer init_commands;
    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    CHECK_VK(vkAllocateCommandBuffers(
        vkDevice, &alloc_info, &init_commands)); // pool will be reset before at
                                                 // begining of first frame

    VkCommandBufferBeginInfo begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    CHECK_VK(vkBeginCommandBuffer(init_commands, &begin_info));

    auto barrier = initialize_evaluation_state();
    VkDependencyInfo depInfo = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    };

    loaded_vkCmdPipelineBarrier2(init_commands, &depInfo);

    CHECK_VK(vkEndCommandBuffer(init_commands));

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &init_commands,
    };
    CHECK_VK(vkQueueSubmit(vkQueue, 1, &submit_info, VK_NULL_HANDLE));
    vkQueueWaitIdle(vkQueue);

    // descriptor & sampler for tile in rendering shader
    {
      VkDescriptorSetAllocateInfo allocInfo = {
          .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
          .descriptorPool = rendering_descriptor_pool,
          .descriptorSetCount = 1,
          .pSetLayouts = &rendering_set_layout,
      };

      vkAllocateDescriptorSets(vkDevice, &allocInfo, &rendering_descriptor_set);

      VkSamplerCreateInfo samplerInfo = {
          .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,

          .magFilter = VK_FILTER_LINEAR,
          .minFilter = VK_FILTER_LINEAR,

          .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,

          .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
          .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
          .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,

          .mipLodBias = 0.0f,
          .anisotropyEnable = VK_FALSE,
          .maxAnisotropy = 1.0f,

          .compareEnable = VK_FALSE,
          .compareOp = VK_COMPARE_OP_ALWAYS,

          .minLod = 0.0f,
          .maxLod = 0.0f, // only one mip level

          .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
          .unnormalizedCoordinates = VK_FALSE,
      };

      vkCreateSampler(vkDevice, &samplerInfo, nullptr, &rendering_tile_sampler);

      VkDescriptorImageInfo imageInfo = {
          .sampler = rendering_tile_sampler,
          .imageView = evaluation_state.tile_image_view,
          .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      };

      VkWriteDescriptorSet write = {
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = rendering_descriptor_set,
          .dstBinding = 0,
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .pImageInfo = &imageInfo};

      vkUpdateDescriptorSets(vkDevice, 1, &write, 0, nullptr);
    }
  }

  // evaluate SDF
  {
    CHECK_VK(vkResetCommandPool(vkDevice, command_pool, 0));
    VkCommandBuffer evaluation_commands;
    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    CHECK_VK(vkAllocateCommandBuffers(
        vkDevice, &alloc_info,
        &evaluation_commands)); // pool will be reset before at
                                // begining of first frame

    VkCommandBufferBeginInfo begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    CHECK_VK(vkBeginCommandBuffer(evaluation_commands, &begin_info));

    vkCmdBindPipeline(evaluation_commands,
                      VkPipelineBindPoint::VK_PIPELINE_BIND_POINT_COMPUTE,
                      evaluation_state.pipeline);

    EvaluationPushConstant evaluation_push_constant = {.tile_pos = {0, 0, 0}};
    vkCmdPushConstants(evaluation_commands, evaluation_state.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, // offset
                       sizeof(EvaluationPushConstant),
                       &evaluation_push_constant);

    vkCmdBindDescriptorSets(evaluation_commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                            evaluation_state.pipeline_layout,
                            0, // firstSet
                            1, // setCount
                            &evaluation_state.descriptor_set, 0, nullptr);

    vkCmdDispatch(evaluation_commands,
                  evaluation_state.tile_image_extent.width / 8,
                  evaluation_state.tile_image_extent.height / 8,
                  evaluation_state.tile_image_extent.depth / 8);

    // transition the tile image to shader read layout
    {
      VkImageMemoryBarrier2 barrier = {
          .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,

          .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,

          .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,

          .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
          .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,

          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,

          .image = evaluation_state.tile_image,
          .subresourceRange =
              {
                  .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                  .baseMipLevel = 0,
                  .levelCount = 1,
                  .baseArrayLayer = 0,
                  .layerCount = 1,
              },
      };
      VkDependencyInfo depInfo = {
          .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
          .imageMemoryBarrierCount = 1,
          .pImageMemoryBarriers = &barrier,
      };

      loaded_vkCmdPipelineBarrier2(evaluation_commands, &depInfo);
    }

    CHECK_VK(vkEndCommandBuffer(evaluation_commands));

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &evaluation_commands,
    };
    CHECK_VK(vkQueueSubmit(vkQueue, 1, &submit_info, VK_NULL_HANDLE));

    // TODO: we should probably not use the queue idle here
    vkQueueWaitIdle(vkQueue);
  }

  // end of initialization

#ifdef XR_USE_PLATFORM_ANDROID
  while (!*android_requests_exit) {
#else
  while (true) {
#endif
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

#ifndef XR_USE_PLATFORM_ANDROID
int main() {
  engine_main();
  return 0;
}
#endif
