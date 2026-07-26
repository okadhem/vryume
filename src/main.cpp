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
#include "glm/ext/vector_float3.hpp"
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
#include <glm/ext/quaternion_float.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>

using namespace glm;

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
static XrSpace xr_space_view;
static XrActionSet gameplayActionSet;
static XrAction move_action;
static XrAction capture_frame_action;
static XrPath leftHandPath;
static XrPath rightHandPath;
static VkPipelineLayout pipelineLayout;
static VkPipeline computePipeline;
static VkFormat color_swapchain_format = VK_FORMAT_R8G8B8A8_UNORM;
// openXR wants depth to represent distance (non normalized).
// TODO: this is not stictly api conform we are using this as a storage image
static VkFormat depth_swapchain_format = VK_FORMAT_D32_SFLOAT;
static PFN_vkCmdPipelineBarrier2KHR loaded_vkCmdPipelineBarrier2;
static uint32_t device_only_memory_type_index;
static VkMemoryPropertyFlags device_only_memory_type_property_flags;
static uint32_t host_visible_memory_type_index;
static VkMemoryPropertyFlags host_visible_memory_type_property_flags;
static uint32_t device_local_host_visible_memory_type_index;
static VkMemoryPropertyFlags
    device_local_host_visible_memory_type_property_flags;
static VkDescriptorPool main_descriptor_pool;
static VkDescriptorSet rendering_descriptor_set;
static RENDERDOC_API_1_4_1 *renderdoc_api = nullptr;
static bool capture_frame = false;

static VkBuffer edit_list_buffer;
static VkDeviceMemory edit_list_memory;
static VkMemoryRequirements edit_list_memory_requirements;

// the transform is the rotation followed by the translation
struct RigidTransform {
  quat rotation;
  vec3 translation;
};

// world space pose of the VR play area, this is the area where the camera
// can move following headset motion
static RigidTransform stage_pose_world;

// space attached to the headset, this is the stage head pose
static RigidTransform head_pose_stage;

// TODO: Keep in sync with the one in glsl until we use a global define or some
// solution
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

// math composition, f . g
RigidTransform compose_transfrom(RigidTransform f, RigidTransform g) {
  return RigidTransform{
      .rotation = f.rotation * g.rotation,
      .translation = (f.rotation * g.translation) + f.translation,
  };
}

quat removePitchRoll(quat q) {
  // headset forward direction
  vec3 forward = q * vec3(0, 0, -1);

  // project onto horizontal plane
  forward.y = 0.0f;
  forward = normalize(forward);

  // compute yaw
  float yaw = atan2(-forward.x, -forward.z);

  // rebuild pure Y rotation
  return angleAxis(yaw, vec3(0, 1, 0));
}
quat inverse_unit_quat(quat q) { return quat(q.w, -q.x, -q.y, -q.z); }

RigidTransform inverse_transform(RigidTransform t) {
  quat inv_rot = inverse_unit_quat(t.rotation);
  return RigidTransform{.rotation = inv_rot,
                        .translation = -(inv_rot * t.translation)};
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
// keep in sync with shader version untill we find a better way to do this.
// layout = std140
struct Edit {
  uint32_t is_removal;     // offset 0  (GLSL bool = 4 bytes)
  int32_t material_id;     // offset 4
  uint32_t primitive_type; // offset 8
  float param0;            // offset 12
  float param1;            // offset 16
  float param2;            // offset 20
  float param3;            // offset 24
  uint32_t _padding0;      // offset 28
  float rotation[4];       // offset 32 (vec4)
  float translation[3];    // offset 48 (vec3)
  uint32_t _padding1;      // offset 60
};
const uint EDIT_PRIMITIVE_SPHERE = 1;
const uint EDIT_PRIMITIVE_BOX = 2;

// layout = std140
struct EditsUniformBuffer {
  uint32_t edit_list_size;
  uint32_t _padding0;
  uint32_t _padding1;
  uint32_t _padding2;
  Edit edit_list[4];
};

VkBufferMemoryBarrier2
buffer_release_compute_writes_to_compute(VkBuffer buffer) {
  return {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,

      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,

      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,

      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,

      .buffer = buffer,
      .offset = 0,
      .size = VK_WHOLE_SIZE,
  };
}

VkImageMemoryBarrier2 image_release_compute_writes_to_compute(VkImage image) {
  return {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,

      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,

      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,

      .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,

      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,

      .image = image,
      .subresourceRange =
          {
              .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
              .baseMipLevel = 0,
              .levelCount = VK_REMAINING_MIP_LEVELS,
              .baseArrayLayer = 0,
              .layerCount = VK_REMAINING_ARRAY_LAYERS,
          },
  };
}

struct SimpleImageCreateInfo {
  VkImageType imageType;
  VkFormat format;
  VkExtent3D extent;
  VkImageUsageFlags usage;
  uint32 memoryTypeIndex;
  VkImageViewType imageViewType;
};
void create_simple_image(SimpleImageCreateInfo simple_info, VkImage *image,
                         VkImageView *image_view, VkDeviceMemory *memory) {
  printf("create_simple_image\n");
  VkImageCreateInfo image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = simple_info.imageType,
      .format = simple_info.format,
      .extent = simple_info.extent,
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = simple_info.usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };

  CHECK_VK(vkCreateImage(vkDevice, &image_info, nullptr, image));
  VkMemoryRequirements memory_requirements;
  vkGetImageMemoryRequirements(vkDevice, *image, &memory_requirements);

  assert(memory_requirements.memoryTypeBits &
         (1u << simple_info.memoryTypeIndex));

  VkMemoryAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = memory_requirements.size,
      .memoryTypeIndex = simple_info.memoryTypeIndex};

  CHECK_VK(vkAllocateMemory(vkDevice, &alloc_info, nullptr, memory));

  vkBindImageMemory(vkDevice, *image, *memory, 0);

  VkImageViewCreateInfo image_view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = *image,
      .viewType = simple_info.imageViewType,
      .format = simple_info.format,
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
  CHECK_VK(vkCreateImageView(vkDevice, &image_view_info, nullptr, image_view));
  printf("created simple image, VkImage:%x\n", *image);
}

struct SimpleBufferCreateInfo {
  VkDeviceSize size;
  VkBufferUsageFlags usage;
  uint32 memoryTypeIndex;
};
void create_simple_buffer(SimpleBufferCreateInfo simple_info, VkBuffer *buffer,
                          VkDeviceMemory *memory,
                          VkMemoryRequirements *memory_requirements) {
  VkBufferCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = simple_info.size,
      .usage = simple_info.usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,

  };
  CHECK_VK(vkCreateBuffer(vkDevice, &create_info, nullptr, buffer));

  vkGetBufferMemoryRequirements(vkDevice, *buffer, memory_requirements);

  assert((*memory_requirements).memoryTypeBits &
         (1u << simple_info.memoryTypeIndex));

  VkMemoryAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = (*memory_requirements).size,
      .memoryTypeIndex = simple_info.memoryTypeIndex,
  };
  CHECK_VK(vkAllocateMemory(vkDevice, &alloc_info, nullptr, memory));

  vkBindBufferMemory(vkDevice, *buffer, *memory,
                     0); // Offset
}

// TODO: these are defined in the shader code, we should find a way to use them
// and redefine.
const int STEP_1_SURFACE =
    0; // when this step is done, all one surface voxels are properly encoded.

const int STEP_2_SURFACE =
    1; // when this step is done, *most* of two surface voxels are properly
       // encoded, precisely, all two surface voxels that
// are away from a three voxel surface, (the line portion away from any
// intersection). other two surface voxel are encoded with *wrong* data namely
// wrong configuration, and invalid sdf values, it is the job of STEP_3_SURFACE
// to fix that.

const int STEP_3_SURFACE =
    2; // when this step is done, all three surface voxels are encoded properly,
       // as well as all the two surface voxels close
// to three surface voxels, left in a wrong state by preious step.

struct MaterialEvaluationPushConstant {
  float tile_pos[3]; // in model space
  int32 step; // one of STEP_XXX, how many sub surface we should encode in this
              // shader run. each step assumes the steps before are done.
  uint32 voxel_texture_width;  // in number of voxels
  uint32 voxel_texture_height; // in number of voxels
  float _pad[2];               // pad to 16 align for the struct
};
struct EvaluationPushConstant {
  float tile_pos[3]; // vec3 alignment=16
  float _pad; // pad to 16 so than next EvaluationPushConstant value in an array
              // starts at the correct alignment
};
struct EvaluationState {
  VkPipelineLayout sdf_pipeline_layout;
  VkPipeline sdf_pipeline;
  VkDescriptorSet sdf_descriptor_set;
  VkDescriptorSetLayout sdf_descriptor_set_layout;

  VkImageView sdf_tile_image_view;
  VkImage sdf_tile_image;
  VkDeviceMemory sdf_tile_image_memory;
  VkExtent3D sdf_tile_image_extent;
  VkSampler sdf_tile_image_sampler;

  VkPipelineLayout materials_pipeline_layout;
  VkPipeline materials_pipeline;
  VkDescriptorSet materials_descriptor_set;
  VkDescriptorSetLayout materials_descriptor_set_layout;

  VkImageView material_info_image_view;
  VkImage material_info_image;
  VkDeviceMemory material_info_image_memory;

  VkBuffer material_sdf_packed;
  VkDeviceMemory material_sdf_packed_memory;
  VkMemoryRequirements material_sdf_packed_memory_requirements;

  VkImageView material_sdf_image_view;
  VkImage material_sdf_image;
  VkDeviceMemory material_sdf_image_memory;
  VkSampler material_sdf_image_sampler;
};

static EvaluationState evaluation_state;
// returns barrier needed to initialize the image before first shader access
void initialize_evaluation_state(VkImageMemoryBarrier2 barriers[3]) {
  // sdf_pipeine and layout
  {
    VkDescriptorSetLayoutBinding sdf_pipeline_bindings[] = {
        {.binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
         .pImmutableSamplers = NULL},

        {.binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
         .pImmutableSamplers = NULL}};

    VkDescriptorSetLayoutCreateInfo evaluation_descriptor_set_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2,
        .pBindings = sdf_pipeline_bindings};

    CHECK_VK(vkCreateDescriptorSetLayout(
        vkDevice, &evaluation_descriptor_set_layout_info, nullptr,
        &evaluation_state.sdf_descriptor_set_layout));

    VkDescriptorSetLayout set_layouts[] = {
        evaluation_state.sdf_descriptor_set_layout};

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
                                    &evaluation_state.sdf_pipeline_layout));

    VkShaderModule shader_module = load_shader_module("evaluation.glsl.spv");

    VkPipelineShaderStageCreateInfo stageInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_module,
        .pName = "main"};

    VkComputePipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stageInfo,
        .layout = evaluation_state.sdf_pipeline_layout};

    CHECK_VK(vkCreateComputePipelines(vkDevice, VK_NULL_HANDLE, 1,
                                      &pipeline_info, NULL,
                                      &evaluation_state.sdf_pipeline));
  }

  // material_pipeline and layout
  {
    VkDescriptorSetLayoutBinding bindings[] = {
        {.binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
         .pImmutableSamplers = NULL},
        {.binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
         .pImmutableSamplers = NULL},
        {.binding = 2,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
         .pImmutableSamplers = NULL},
        {.binding = 3,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
         .pImmutableSamplers = NULL}};

    VkDescriptorSetLayoutCreateInfo descriptor_set_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 4,
        .pBindings = bindings};

    CHECK_VK(vkCreateDescriptorSetLayout(
        vkDevice, &descriptor_set_layout_info, nullptr,
        &evaluation_state.materials_descriptor_set_layout));

    VkDescriptorSetLayout set_layouts[] = {
        evaluation_state.materials_descriptor_set_layout};

    VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(MaterialEvaluationPushConstant),
    };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = set_layouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range};

    CHECK_VK(
        vkCreatePipelineLayout(vkDevice, &pipelineLayoutInfo, nullptr,
                               &evaluation_state.materials_pipeline_layout));

    VkShaderModule shader_module =
        load_shader_module("material_evaluation.glsl.spv");

    VkPipelineShaderStageCreateInfo stageInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_module,
        .pName = "main"};

    VkComputePipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stageInfo,
        .layout = evaluation_state.materials_pipeline_layout};

    CHECK_VK(vkCreateComputePipelines(vkDevice, VK_NULL_HANDLE, 1,
                                      &pipeline_info, NULL,
                                      &evaluation_state.materials_pipeline));
  }

  {
    evaluation_state.sdf_tile_image_extent = {
        .width = 1000, .height = 1000, .depth = 1000};

    create_simple_image(
        SimpleImageCreateInfo{
            .imageType = VK_IMAGE_TYPE_3D,
            .format = VK_FORMAT_R8_UNORM,
            // TODO image size must be divisible by shader group size, we need a
            // way to make this less error prone
            .extent = evaluation_state.sdf_tile_image_extent,
            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            .imageViewType = VK_IMAGE_VIEW_TYPE_3D,
            .memoryTypeIndex = device_only_memory_type_index,
        },
        &evaluation_state.sdf_tile_image, &evaluation_state.sdf_tile_image_view,
        &evaluation_state.material_info_image_memory);

    VkExtent3D material_info_extent = {
        .width = evaluation_state.sdf_tile_image_extent.width - 1,
        .height = evaluation_state.sdf_tile_image_extent.height - 1,
        .depth = evaluation_state.sdf_tile_image_extent.depth - 1,
    };
    create_simple_image(
        SimpleImageCreateInfo{
            .imageType = VK_IMAGE_TYPE_3D,
            .format = VK_FORMAT_R8_UINT,
            // TODO image size must be divisible by shader group size, we
            // need a way to make this less error prone
            .extent = material_info_extent,
            .usage = VK_IMAGE_USAGE_STORAGE_BIT,
            .imageViewType = VK_IMAGE_VIEW_TYPE_3D,
            .memoryTypeIndex = device_only_memory_type_index,
        },
        &evaluation_state.material_info_image,
        &evaluation_state.material_info_image_view,
        &evaluation_state.material_info_image_memory);

    create_simple_buffer(
        SimpleBufferCreateInfo{
            .size = evaluation_state.sdf_tile_image_extent.width *
                    evaluation_state.sdf_tile_image_extent.height *
                    evaluation_state.sdf_tile_image_extent.depth,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .memoryTypeIndex =
                host_visible_memory_type_index, // only GPU will use this, yet
                                                // we do this to get a page in
                                                // RAM because of memory
                                                // pressure
        },
        &evaluation_state.material_sdf_packed,
        &evaluation_state.material_sdf_packed_memory,
        &evaluation_state.material_sdf_packed_memory_requirements);

    create_simple_image(
        SimpleImageCreateInfo{
            .imageType = VK_IMAGE_TYPE_3D,
            .format = VK_FORMAT_R4G4_UNORM_PACK8,
            .extent = evaluation_state.sdf_tile_image_extent,
            .usage =
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            .imageViewType = VK_IMAGE_VIEW_TYPE_3D,
            .memoryTypeIndex = device_only_memory_type_index,
        },
        &evaluation_state.material_sdf_image,
        &evaluation_state.material_sdf_image_view,
        &evaluation_state.material_sdf_image_memory);

    VkSamplerCreateInfo sdf_tile_image_sampler_info = {
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

    vkCreateSampler(vkDevice, &sdf_tile_image_sampler_info, nullptr,
                    &evaluation_state.sdf_tile_image_sampler);

    VkSamplerCreateInfo material_sdf_image_sampler_info = {
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
    vkCreateSampler(vkDevice, &material_sdf_image_sampler_info, nullptr,
                    &evaluation_state.material_sdf_image_sampler);
  }

  // descriptors for sdf_pipeline
  {
    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = main_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &evaluation_state.sdf_descriptor_set_layout,
    };

    CHECK_VK(vkAllocateDescriptorSets(vkDevice, &alloc_info,
                                      &evaluation_state.sdf_descriptor_set));

    VkDescriptorImageInfo image_info = {
        .sampler = VK_NULL_HANDLE,
        .imageView = evaluation_state.sdf_tile_image_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL, // the layout our image will be
                                                // at when used by shader
    };

    VkDescriptorBufferInfo buffer_info = {
        .buffer = edit_list_buffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };

    VkWriteDescriptorSet writes[] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = evaluation_state.sdf_descriptor_set,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &image_info,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = evaluation_state.sdf_descriptor_set,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &buffer_info,
        },
    };

    vkUpdateDescriptorSets(vkDevice, 2, writes, 0, nullptr);
  }

  // descriptors for materials pipeline
  {
    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = main_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &evaluation_state.materials_descriptor_set_layout,
    };

    CHECK_VK(vkAllocateDescriptorSets(
        vkDevice, &alloc_info, &evaluation_state.materials_descriptor_set));

    VkDescriptorImageInfo material_image_info = {
        .sampler = VK_NULL_HANDLE,
        .imageView = evaluation_state.material_info_image_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL, // the layout our image will be
                                                // at when used by shader
    };

    VkDescriptorImageInfo sdf_tile_image_info = {
        .sampler = evaluation_state.sdf_tile_image_sampler,
        .imageView = evaluation_state.sdf_tile_image_view,
        .imageLayout =
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, // the layout our image
                                                      // will be at when used by
                                                      // shader
    };

    VkDescriptorBufferInfo edit_list_buffer_info = {
        .buffer = edit_list_buffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };

    VkDescriptorBufferInfo material_sdf_packed_buffer_info = {
        .buffer = evaluation_state.material_sdf_packed,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };

    VkWriteDescriptorSet writes[] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = evaluation_state.materials_descriptor_set,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &edit_list_buffer_info,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = evaluation_state.materials_descriptor_set,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &material_image_info,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = evaluation_state.materials_descriptor_set,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &material_sdf_packed_buffer_info,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = evaluation_state.materials_descriptor_set,
            .dstBinding = 3,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &sdf_tile_image_info,
        },
    };

    vkUpdateDescriptorSets(vkDevice, 4, writes, 0, nullptr);
  }
  barriers[0] = {
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

      .image = evaluation_state.sdf_tile_image,
      .subresourceRange =
          {
              .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
              .baseMipLevel = 0,
              .levelCount = 1,
              .baseArrayLayer = 0,
              .layerCount = 1,
          },
  };
  barriers[1] = {
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

      .image = evaluation_state.material_info_image,
      .subresourceRange =
          {
              .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
              .baseMipLevel = 0,
              .levelCount = 1,
              .baseArrayLayer = 0,
              .layerCount = 1,
          },
  };
  barriers[2] = {
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

      .image = evaluation_state.material_sdf_image,
      .subresourceRange =
          {
              .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
              .baseMipLevel = 0,
              .levelCount = 1,
              .baseArrayLayer = 0,
              .layerCount = 1,
          },
  };
}

struct InputState {
  XrActionStateVector2f right_move;
  XrActionStateVector2f left_move;
  XrActionStateBoolean capture_frame;
  bool right_move_horizental_triggered;
  bool right_move_vertical_triggered;
};
// after how much is the continuous input considered triggered
constexpr float input_trigger_threshold = 0.8;
static InputState inputs = {};
static InputState previous_inputs = {};
struct RenderingPushConstant {
  float camera_orientation_world[4];
  float camera_position_world[3];
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
  float _pad4[2];
  uint32
      tile_extent_voxels[3]; // number of voxels in each direction, starts at 96
  float _pad5;               // to align struct to 16
};
static_assert(sizeof(RenderingPushConstant) == 112);
static_assert(sizeof(RenderingPushConstant) % 16 == 0);
static_assert(sizeof(RenderingPushConstant) < 128, "PushConstant limit");

static bool is_xr_session_running = false;
static bool should_run_framecycle = false;
static XrTime display_time;              // in ns
static XrTime previous_display_time = 0; // in ns
static XrDuration frame_duration;        // in ns
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
    previous_display_time = display_time;
    display_time = frame_state.predictedDisplayTime;
    if (previous_display_time == 0)
      previous_display_time = display_time;

    // TODO: there is a frame duration in the frame_state, check that
    frame_duration = display_time - previous_display_time;
  }

  // TODO
  if (!frame_state.shouldRender) {
    LOGI("frame_state.shouldRender is false\n");
    //   return 0;
  }

  // should we ? assert(_out_view_count == viewCount);

  XrFrameBeginInfo frame_begin_info = {
      .type = XR_TYPE_FRAME_BEGIN_INFO,
  };
  CHECK_XR(xrBeginFrame(session, &frame_begin_info));

  CHECK_VK(
      vkWaitForFences(vkDevice, viewCount, fence_exec, VK_TRUE, UINT64_MAX));

  bool frame_capture_ongoing = false;
  if (renderdoc_api && capture_frame) {
    renderdoc_api->StartFrameCapture(NULL, NULL);
    frame_capture_ongoing = true;
  }

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

  // stage_view_pose
  {
    XrSpaceLocation location = {.type = XR_TYPE_SPACE_LOCATION};
    xrLocateSpace(xr_space_view, xr_space_stage,
                  frame_state.predictedDisplayTime, &location);
    auto o = location.pose.orientation;
    auto p = location.pose.position;
    head_pose_stage = RigidTransform{
        .rotation = quat(o.w, o.x, o.y, o.z),
        .translation = vec3(p.x, p.y, p.z),
    };
  }

  // previous_inputs
  // inputs
  {
    previous_inputs = inputs;
    XrActiveActionSet activeSet = {
        .actionSet = gameplayActionSet,
    };

    XrActionsSyncInfo syncInfo = {
        .type = XR_TYPE_ACTIONS_SYNC_INFO,
        .countActiveActionSets = 1,
        .activeActionSets = &activeSet,
    };
    auto r = xrSyncActions(session, &syncInfo);
    if (r < 0) {
      LOGE("SyncActions returned an error %d\n", r);
    }

    XrActionStateVector2f right_move_state = {
        .type = XR_TYPE_ACTION_STATE_VECTOR2F,
    };

    XrActionStateGetInfo right_get_action_info = {
        .type = XR_TYPE_ACTION_STATE_GET_INFO,
        .action = move_action,
        .subactionPath = rightHandPath,
    };

    CHECK_XR(xrGetActionStateVector2f(session, &right_get_action_info,
                                      &right_move_state));

    XrActionStateVector2f left_move_state = {
        .type = XR_TYPE_ACTION_STATE_VECTOR2F,
    };

    XrActionStateGetInfo left_get_action_info = {
        .type = XR_TYPE_ACTION_STATE_GET_INFO,
        .action = move_action,
        .subactionPath = leftHandPath,
    };

    CHECK_XR(xrGetActionStateVector2f(session, &left_get_action_info,
                                      &left_move_state));

    XrActionStateBoolean capture_frame_state = {
        .type = XR_TYPE_ACTION_STATE_BOOLEAN,
    };

    XrActionStateGetInfo capture_frame_state_info = {
        .type = XR_TYPE_ACTION_STATE_GET_INFO,
        .action = capture_frame_action,
        .subactionPath = rightHandPath,
    };

    CHECK_XR(xrGetActionStateBoolean(session, &capture_frame_state_info,
                                     &capture_frame_state));

    inputs = {
        .right_move = right_move_state,
        .left_move = left_move_state,
        .capture_frame = capture_frame_state,
        .right_move_horizental_triggered =
            abs(right_move_state.currentState.x) > input_trigger_threshold,
        .right_move_vertical_triggered =
            abs(right_move_state.currentState.y) > input_trigger_threshold,
    };
  }

  // responding to inputs
  {
    constexpr float camera_translation_speed = 1e-9; // in m/ns

    RigidTransform head_vertical_pose_stage = {
        .rotation = removePitchRoll(head_pose_stage.rotation),
        .translation = head_pose_stage.translation,
    };

    RigidTransform new_pose = {.rotation = quat(1, 0, 0, 0),
                               .translation = vec3(0)};

    if (inputs.right_move.isActive && inputs.right_move_horizental_triggered &&
        !previous_inputs.right_move_horizental_triggered) {

      if (inputs.right_move.currentState.x > 0) {
        auto minus_30_y = quat(0.9659258, 0, -0.2588192, 0);
        new_pose.rotation = minus_30_y;
      } else {
        auto plus_30_y = quat(0.9659258, 0, 0.2588192, 0);
        new_pose.rotation = plus_30_y;
      }
    }

    vec3 translation = vec3(0);
    if (inputs.right_move.isActive && inputs.right_move_vertical_triggered) {
      auto displacement = camera_translation_speed * frame_duration;
      translation +=
          vec3(0, inputs.right_move.currentState.y * displacement, 0);
    }

    if (inputs.left_move.isActive) {
      auto displacement = camera_translation_speed * frame_duration;
      translation += vec3(inputs.left_move.currentState.x * displacement, 0,
                          -(inputs.left_move.currentState.y * displacement));
    }

    new_pose.translation = translation;

    stage_pose_world = compose_transfrom(
        stage_pose_world,
        compose_transfrom(
            head_vertical_pose_stage,
            compose_transfrom(new_pose,
                              inverse_transform(head_vertical_pose_stage))));

    if (inputs.capture_frame.currentState) {
      capture_frame = true; // this will start the capture next frame
    }
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

    RigidTransform eye_pose_world;
    {
      auto o = views[i].pose.orientation;
      auto p = views[i].pose.position;

      RigidTransform eye_pose_stage = {
          .rotation = quat(o.w, o.x, o.y, o.z),
          .translation = vec3(p.x, p.y, p.z),
      };

      eye_pose_world = compose_transfrom(stage_pose_world, eye_pose_stage);
    }

    printf("eye_pos_world: %f,%f,%f\n", eye_pose_world.translation.x,
           eye_pose_world.translation.y, eye_pose_world.translation.z);
    // eye_pose_world.translation = vec3(0.859198, 1.173761, 1.817214);
    // printf("eye_pos_world: %f,%f,%f,%f\n", eye_pose_world.rotation.x,
    //        eye_pose_world.rotation.y, eye_pose_world.rotation.z,
    //        eye_pose_world.rotation.w);
    // eye_pose_world.rotation = quat(-0.96533, 0.17188, 0.1964, -0.00273);

    RenderingPushConstant rendering_push_constant = {
        .camera_orientation_world = {eye_pose_world.rotation.x,
                                     eye_pose_world.rotation.y,
                                     eye_pose_world.rotation.z,
                                     eye_pose_world.rotation.w},
        .camera_position_world = {eye_pose_world.translation.x,
                                  eye_pose_world.translation.y,
                                  eye_pose_world.translation.z},
        .tanLeft = (float)tan(views[i].fov.angleLeft),
        .tanRight = (float)tan(views[i].fov.angleRight),
        .tanUp = (float)tan(views[i].fov.angleUp),
        .tanDown = (float)tan(views[i].fov.angleDown),
        .tile_pos = {0.0, 0.0, 0.0},
        .tile_extent =
            {
                (float)((evaluation_state.sdf_tile_image_extent.width - 1) *
                        inter_sample_distance),
                (float)((evaluation_state.sdf_tile_image_extent.height - 1) *
                        inter_sample_distance),
                (float)((evaluation_state.sdf_tile_image_extent.depth - 1) *
                        inter_sample_distance),
            },
        .resolution = {(float)color_swapchains[i].width,
                       (float)color_swapchains[i].height},
        .tile_extent_voxels =
            {
                evaluation_state.sdf_tile_image_extent.width - 1,
                evaluation_state.sdf_tile_image_extent.height - 1,
                evaluation_state.sdf_tile_image_extent.depth - 1,
            },

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
              .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,

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

  if (renderdoc_api && frame_capture_ongoing) {
    vkQueueWaitIdle(vkQueue);
    renderdoc_api->EndFrameCapture(NULL, NULL);
    frame_capture_ongoing = false;
    capture_frame = false;
  }

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

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(vkPhysicalDevice, &props);

    printf("GPU name: %s\n", props.deviceName);
    printf("Vendor ID: 0x%x\n", props.vendorID);
    printf("Device ID: 0x%x\n", props.deviceID);
    printf("Driver version: %u\n", props.driverVersion);
    printf("API version: %u.%u.%u\n", VK_VERSION_MAJOR(props.apiVersion),
           VK_VERSION_MINOR(props.apiVersion),
           VK_VERSION_PATCH(props.apiVersion));
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

  // host_visible_memory_type_index
  // host_visible_memory_type_property_flags
  {
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(vkPhysicalDevice, &memory_properties);

    bool found_memory_type = false;
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
      VkMemoryPropertyFlags flags =
          memory_properties.memoryTypes[i].propertyFlags;
      if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
          !(flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
        host_visible_memory_type_index = i;
        host_visible_memory_type_property_flags = flags;
        found_memory_type = true;
        break;
      }
    }
    assert(found_memory_type);
  }

  // device_local_host_visible_memory_type_index
  // device_local_host_visible_memory_type_property_flags
  {
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(vkPhysicalDevice, &memory_properties);

    bool found_memory_type = false;
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
      VkMemoryPropertyFlags flags =
          memory_properties.memoryTypes[i].propertyFlags;
      if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
          (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
        device_local_host_visible_memory_type_index = i;
        device_local_host_visible_memory_type_property_flags = flags;
        found_memory_type = true;
        break;
      }
    }
    assert(found_memory_type);
    printf("memory id: %d\n", device_local_host_visible_memory_type_index);
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

    XrReferenceSpaceCreateInfo xr_reference_space_view_create_info = {
        .type = XR_TYPE_REFERENCE_SPACE_CREATE_INFO,
        .referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW,
        .poseInReferenceSpace = xr_pose_identity,
    };
    CHECK_XR(xrCreateReferenceSpace(
        session, &xr_reference_space_view_create_info, &xr_space_view));
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

    VkDescriptorSetLayoutBinding rendering_descriptor_set_bindings[] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };

    VkDescriptorSetLayoutCreateInfo rendering_set_layout_create_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = rendering_descriptor_set_bindings};

    vkCreateDescriptorSetLayout(vkDevice, &rendering_set_layout_create_info,
                                nullptr, &rendering_set_layout);

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

  // main_descriptor_pool
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
                                    &main_descriptor_pool));
  }

  // openxr actions (input system)
  {

    XrActionSetCreateInfo actionSetInfo = {
        .type = XR_TYPE_ACTION_SET_CREATE_INFO,
        .actionSetName = "gameplay",
        .localizedActionSetName = "Gameplay",
        .priority = 0,
    };

    CHECK_XR(xrCreateActionSet(instance, &actionSetInfo, &gameplayActionSet));

    CHECK_XR(xrStringToPath(instance, "/user/hand/left", &leftHandPath));
    CHECK_XR(xrStringToPath(instance, "/user/hand/right", &rightHandPath));

    XrPath handSubactionPaths[] = {leftHandPath, rightHandPath};

    XrActionCreateInfo move_action_info = {
        .type = XR_TYPE_ACTION_CREATE_INFO,
        .actionName = "move",
        .actionType = XR_ACTION_TYPE_VECTOR2F_INPUT,
        .countSubactionPaths = 2,
        .subactionPaths = handSubactionPaths,
        .localizedActionName = "Move",
    };

    CHECK_XR(
        xrCreateAction(gameplayActionSet, &move_action_info, &move_action));

    XrActionCreateInfo capture_frame_action_info = {
        .type = XR_TYPE_ACTION_CREATE_INFO,
        .actionName = "capture_frame",
        .actionType = XR_ACTION_TYPE_BOOLEAN_INPUT,
        .countSubactionPaths = 2,
        .subactionPaths = handSubactionPaths,
        .localizedActionName = "Capture Frame",
    };

    CHECK_XR(xrCreateAction(gameplayActionSet, &capture_frame_action_info,
                            &capture_frame_action));

    {
      XrPath thumbstick_left_path;
      CHECK_XR(xrStringToPath(instance, "/user/hand/left/input/thumbstick",
                              &thumbstick_left_path));

      XrPath thumbstick_right_path;
      CHECK_XR(xrStringToPath(instance, "/user/hand/right/input/thumbstick",
                              &thumbstick_right_path));

      XrPath button_b_path;
      CHECK_XR(xrStringToPath(instance, "/user/hand/right/input/b/click",
                              &button_b_path));

      XrActionSuggestedBinding bindings[] = {
          {
              .action = move_action,
              .binding = thumbstick_left_path,
          },
          {
              .action = move_action,
              .binding = thumbstick_right_path,
          },
          {
              .action = capture_frame_action,
              .binding = button_b_path,
          },
      };

      XrPath interactionProfilePath;
      CHECK_XR(xrStringToPath(instance,
                              "/interaction_profiles/oculus/touch_controller",
                              &interactionProfilePath));

      XrInteractionProfileSuggestedBinding suggested = {
          .type = XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING,
          .interactionProfile = interactionProfilePath,
          .countSuggestedBindings = 3,
          .suggestedBindings = bindings,
      };

      CHECK_XR(xrSuggestInteractionProfileBindings(instance, &suggested));
    }

    XrSessionActionSetsAttachInfo attachInfo = {
        .type = XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO,
        .countActionSets = 1,
        .actionSets = &gameplayActionSet,
    };

    CHECK_XR(xrAttachSessionActionSets(session, &attachInfo));
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

    // edit_list_buffer
    // edit_list_memory
    {
      create_simple_buffer(
          SimpleBufferCreateInfo{
              .size = sizeof(EditsUniformBuffer),
              .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
              .memoryTypeIndex = device_local_host_visible_memory_type_index,
          },
          &edit_list_buffer, &edit_list_memory, &edit_list_memory_requirements);

      EditsUniformBuffer buf = {
          .edit_list_size = 3,
          .edit_list =
              {
                  {
                      .is_removal = false,
                      .material_id = 1,
                      .primitive_type = EDIT_PRIMITIVE_SPHERE,
                      .param0 = 0.25,
                      .translation = {1.0, 0.65, 1.1},
                  },
                  {
                      .is_removal = false,
                      .material_id = 2,
                      .primitive_type = EDIT_PRIMITIVE_SPHERE,
                      .param0 = 0.25,
                      .translation = {1.0, 1.0, 1.1},
                  },
                  {
                      .is_removal = false,
                      .material_id = 3,
                      .primitive_type = EDIT_PRIMITIVE_SPHERE,
                      .param0 = 0.25,
                      .translation = {1.3, 0.75, 1.1},
                  },
              },
      };

      void *mapped = nullptr;
      CHECK_VK(vkMapMemory(vkDevice, edit_list_memory, 0,
                           edit_list_memory_requirements.size, 0, &mapped));

      memcpy(mapped, &buf, sizeof(buf));

      VkMappedMemoryRange range{
          .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
          .memory = edit_list_memory,
          .offset = 0,
          .size = VK_WHOLE_SIZE,
      };
      CHECK_VK(vkFlushMappedMemoryRanges(vkDevice, 1, &range));

      vkUnmapMemory(vkDevice, edit_list_memory);
    }

    CHECK_VK(vkBeginCommandBuffer(init_commands, &begin_info));
    VkImageMemoryBarrier2 barriers[3];
    initialize_evaluation_state(barriers);
    VkDependencyInfo depInfo = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 3, // TODO find a way less error prone
        .pImageMemoryBarriers = barriers,
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

    // descriptors for tile realted resources in rendering shader
    {
      VkDescriptorSetAllocateInfo allocInfo = {
          .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
          .descriptorPool = main_descriptor_pool,
          .descriptorSetCount = 1,
          .pSetLayouts = &rendering_set_layout,
      };

      vkAllocateDescriptorSets(vkDevice, &allocInfo, &rendering_descriptor_set);

      VkDescriptorImageInfo imageInfo = {
          .sampler = evaluation_state.sdf_tile_image_sampler,
          .imageView = evaluation_state.sdf_tile_image_view,
          .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      };
      VkDescriptorImageInfo material_info_image_info = {
          .imageView = evaluation_state.material_info_image_view,
          .imageLayout =
              VK_IMAGE_LAYOUT_GENERAL, // even if we readonly, READ_ONLY_OPTIMAL
                                       // is for sampled images
      };
      VkDescriptorImageInfo material_sdf_image_info = {
          .sampler = evaluation_state.material_sdf_image_sampler,
          .imageView = evaluation_state.material_sdf_image_view,
          .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      };
      VkWriteDescriptorSet writes[] = {
          {
              .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
              .dstSet = rendering_descriptor_set,
              .dstBinding = 0,
              .dstArrayElement = 0,
              .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              .pImageInfo = &imageInfo,
          },
          {
              .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
              .dstSet = rendering_descriptor_set,
              .dstBinding = 1,
              .dstArrayElement = 0,
              .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              .pImageInfo = &material_info_image_info,
          },
          {
              .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
              .dstSet = rendering_descriptor_set,
              .dstBinding = 2,
              .dstArrayElement = 0,
              .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              .pImageInfo = &material_sdf_image_info,
          },
      };

      vkUpdateDescriptorSets(vkDevice, 3, writes, 0, nullptr);
    }
  }

  // evaluate SDF
  {

    if (renderdoc_api) {
      // renderdoc_api->StartFrameCapture(NULL, NULL);
    }

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
                      evaluation_state.sdf_pipeline);

    EvaluationPushConstant evaluation_push_constant = {.tile_pos = {0, 0, 0}};
    vkCmdPushConstants(
        evaluation_commands, evaluation_state.sdf_pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0, // offset
        sizeof(EvaluationPushConstant), &evaluation_push_constant);

    vkCmdBindDescriptorSets(evaluation_commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                            evaluation_state.sdf_pipeline_layout,
                            0, // firstSet
                            1, // setCount
                            &evaluation_state.sdf_descriptor_set, 0, nullptr);

    vkCmdDispatch(evaluation_commands,
                  evaluation_state.sdf_tile_image_extent.width / 8,
                  evaluation_state.sdf_tile_image_extent.height / 8,
                  evaluation_state.sdf_tile_image_extent.depth / 8);

    // transition the tile image to shader read layout
    {
      VkImageMemoryBarrier2 barrier = {
          .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,

          .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,

          .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,

          .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
          .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,

          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,

          .image = evaluation_state.sdf_tile_image,
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

    // material evaluation commands
    {
      uint32 texture_voxel_width =
          evaluation_state.sdf_tile_image_extent.width - 1;
      uint32 texture_voxel_height =
          evaluation_state.sdf_tile_image_extent.height - 1;
      uint32 texture_voxel_depth =
          evaluation_state.sdf_tile_image_extent.depth - 1;

      vkCmdBindPipeline(evaluation_commands,
                        VkPipelineBindPoint::VK_PIPELINE_BIND_POINT_COMPUTE,
                        evaluation_state.materials_pipeline);

      vkCmdBindDescriptorSets(
          evaluation_commands, VK_PIPELINE_BIND_POINT_COMPUTE,
          evaluation_state.materials_pipeline_layout,
          0, // firstSet
          1, // setCount
          &evaluation_state.materials_descriptor_set, 0, nullptr);

      // step 1
      MaterialEvaluationPushConstant material_push_constants = {
          .step = STEP_1_SURFACE,
          .tile_pos = {0, 0, 0},
          .voxel_texture_width = texture_voxel_width,
          .voxel_texture_height = texture_voxel_height,
      };

      vkCmdPushConstants(
          evaluation_commands, evaluation_state.materials_pipeline_layout,
          VK_SHADER_STAGE_COMPUTE_BIT,
          0, // offset
          sizeof(MaterialEvaluationPushConstant), &material_push_constants);

      vkCmdDispatch(evaluation_commands, texture_voxel_width / 9,
                    texture_voxel_height / 9, texture_voxel_depth / 9);
      {
        VkImageMemoryBarrier2 img_barrier =
            image_release_compute_writes_to_compute(
                evaluation_state.material_info_image);
        VkBufferMemoryBarrier2 buf_barrier =
            buffer_release_compute_writes_to_compute(
                evaluation_state.material_sdf_packed);

        VkDependencyInfo depInfo = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &img_barrier,
            .pBufferMemoryBarriers = &buf_barrier,
            .bufferMemoryBarrierCount = 1,
        };
        loaded_vkCmdPipelineBarrier2(evaluation_commands, &depInfo);
      }

      // step 2
      material_push_constants.step = STEP_2_SURFACE;
      vkCmdPushConstants(
          evaluation_commands, evaluation_state.materials_pipeline_layout,
          VK_SHADER_STAGE_COMPUTE_BIT,
          0, // offset
          sizeof(MaterialEvaluationPushConstant), &material_push_constants);

      vkCmdDispatch(evaluation_commands, texture_voxel_width / 9,
                    texture_voxel_height / 9, texture_voxel_depth / 9);
      {
        VkImageMemoryBarrier2 img_barrier =
            image_release_compute_writes_to_compute(
                evaluation_state.material_info_image);
        VkBufferMemoryBarrier2 buf_barrier =
            buffer_release_compute_writes_to_compute(
                evaluation_state.material_sdf_packed);

        VkDependencyInfo depInfo = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &img_barrier,
            .pBufferMemoryBarriers = &buf_barrier,
            .bufferMemoryBarrierCount = 1,
        };
        loaded_vkCmdPipelineBarrier2(evaluation_commands, &depInfo);
      }
      // step 3
      material_push_constants.step = STEP_3_SURFACE;
      vkCmdPushConstants(
          evaluation_commands, evaluation_state.materials_pipeline_layout,
          VK_SHADER_STAGE_COMPUTE_BIT,
          0, // offset
          sizeof(MaterialEvaluationPushConstant), &material_push_constants);

      vkCmdDispatch(evaluation_commands, texture_voxel_width / 9,
                    texture_voxel_height / 9, texture_voxel_depth / 9);

      // prepare the buffer and image for the copy
      {
        // before any copy operation writes this image, perform the layout
        // transition.
        VkImageMemoryBarrier2 image_barrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
            .srcAccessMask = 0,
            .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .image = evaluation_state.material_sdf_image,
            .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .subresourceRange.levelCount = 1,
            .subresourceRange.layerCount = 1,
        };

        VkBufferMemoryBarrier2 buffer_barrier{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
            .buffer = evaluation_state.material_sdf_packed,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        };

        VkDependencyInfo depInfo = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &image_barrier,
            .pBufferMemoryBarriers = &buffer_barrier,
            .bufferMemoryBarrierCount = 1,
        };
        loaded_vkCmdPipelineBarrier2(evaluation_commands, &depInfo);
      }

      VkBufferImageCopy region{
          region.bufferOffset = 0,
          // 0 = tightly packed
          .bufferRowLength = 0,
          .bufferImageHeight = 0,
          .imageSubresource =
              {
                  .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                  .mipLevel = 0,
                  .baseArrayLayer = 0,
                  .layerCount = 1,
              },
          .imageOffset = {0, 0, 0},
          .imageExtent = evaluation_state.sdf_tile_image_extent,
      };

      vkCmdCopyBufferToImage(evaluation_commands,
                             evaluation_state.material_sdf_packed,
                             evaluation_state.material_sdf_image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

      // make material images available for later compute shader renderer work
      {
        VkImageMemoryBarrier2 barriers[] = {
            {
                // material_sdf_barrier
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .image = evaluation_state.material_sdf_image,
                .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .subresourceRange.levelCount = 1,
                .subresourceRange.layerCount = 1,
            },
            {
                // material_info_barrier
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .image = evaluation_state.material_info_image,
                .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .subresourceRange.levelCount = 1,
                .subresourceRange.layerCount = 1,
            },
        };
        VkDependencyInfo depInfo = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 2,
            .pImageMemoryBarriers = barriers,
        };
        loaded_vkCmdPipelineBarrier2(evaluation_commands, &depInfo);
      }
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

    if (renderdoc_api) {
      // renderdoc_api->EndFrameCapture(NULL, NULL);
    }
  }

  stage_pose_world = RigidTransform{
      .rotation = glm::quat(1.0, 0.0, 0.0, 0.0),
      .translation = glm::vec3(0.0, 0.0, 0.0),
  };

  // end of initialization

#ifdef XR_USE_PLATFORM_ANDROID
  while (!*android_requests_exit) {
#else
  while (true) {
#endif
    tick();
  }

  LOGI("last frame_duration:%ld", frame_duration);

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

// TODO:
// Pools, OK
// push constants: main.glsl and evalulation did not change - all done
// writing the edit list -- need to proper scene -- done
// commands: missing step 3 -- done
// init the material info image with a proper "null" value -- done

// at buffer copy, the buffer is zero