#include <stdint.h>
#include <math.h>
#include <stdio.h>

#include <math.h>
#include <algorithm>
typedef struct {
  float x, y;
} vec2;
typedef struct {
  float d00, d10, d01, d11;
} SdfCell;
static vec2 normalize(vec2 v) {
  float len = sqrtf(v.x * v.x + v.y * v.y);
  vec2 r = {v.x / len, v.y / len};
  return r;
}

static float sdLine(vec2 p, vec2 p0, vec2 dir) {
  vec2 d = normalize(dir);
  vec2 n = {-d.y, d.x}; // perpendicular normal
  return (p.x - p0.x) * n.x + (p.y - p0.y) * n.y;
}

static SdfCell computeCellSDF(vec2 c, float h, vec2 linePoint, vec2 lineDir) {
  SdfCell out;

  vec2 p00 = c;
  vec2 p10 = (vec2){c.x + h, c.y};
  vec2 p01 = (vec2){c.x, c.y + h};
  vec2 p11 = (vec2){c.x + h, c.y + h};

  out.d00 = sdLine(p00, linePoint, lineDir);
  out.d10 = sdLine(p10, linePoint, lineDir);
  out.d01 = sdLine(p01, linePoint, lineDir);
  out.d11 = sdLine(p11, linePoint, lineDir);

  return out;
}

static float max_distance = 0.002;
static float min_distance = -0.002;

float mix(float x, float y, float a) { return x * (1.0f - a) + y * a; }

// inverse of glsl mix function
// given x in [a,b] return the interpolation parameter in range [0, 1]
// it must be the case that: a < b and x in [a,b]
float inverse_mix(float a, float b, float x) { return (x - a) / (b - a); }

float encode_distance(float distance) {
  float clamped = std::clamp(distance, min_distance, max_distance);
  return inverse_mix(min_distance, max_distance, clamped);
}

float decode_distance(float sampled_distance) {
  return mix(min_distance, max_distance, sampled_distance);
}

float gpu_int8_to_norm_float(uint8_t i) { return i / 255.0; }

// f must be normalized
uint8_t gpu_norm_float_to_int8(float f) { return round(f * 255.0); }

uint8_t encode_distance_4bits(float distance) {
  // Clamp
  if (distance > max_distance)
    distance = max_distance;
  if (distance < -max_distance)
    distance = -max_distance;

  // Scale to [-8, 7]
  int value = (int)lroundf(distance * 7.0f / max_distance);

  // Safety clamp
  if (value > 7)
    value = 7;
  if (value < -8)
    value = -8;

  // Convert to 4-bit two's complement representation
  return (uint8_t)value & 0xF;
}
float decode_distance_4bits(uint8_t bits) {
  bits &= 0xF;

  int value;

  if (bits & 0x8)
    value = bits - 16;
  else
    value = bits;

  return (float)value * max_distance / 7.0f;
}
// fx, fy are between 0 and 1
// texels are in this order: (x0,y0) / (x1,y0) / (x0,y1) / (x1,y1)
float sample(float fx, float fy, float texels[4]) {
  // v=v00​(1−fx​)(1−fy​)+v10​fx​(1−fy​)+v01​(1−fx​)fy​+v11​fx​fy
  return texels[0] * (1 - fx) * (1 - fy) + texels[1] * fx * (1 - fy) +
         texels[2] * (1 - fx) * fy + texels[3] * fx * fy;
}

int main() {
  vec2 cell = {0.0f, 0.0f};
  float h = 0.002f;
  vec2 linePoint1 = {0.0f, 0.0f};
  vec2 lineDir1 = {h, h};
  // SdfCell cell_sdf1 = computeCellSDF(cell, h, linePoint1, lineDir1);
  SdfCell cell_sdf2 = computeCellSDF(cell, h, linePoint1, lineDir1);

  vec2 linePoint2 = {0.001f, 0.0f};
  vec2 lineDir2 = {0.0f, h};
  // SdfCell cell_sdf2 = computeCellSDF(cell, h, linePoint2, lineDir2);
  SdfCell cell_sdf1 = computeCellSDF(cell, h, linePoint2, lineDir2);

  // p00 ---- p10
  //  |        |
  // p01 ---- p11
  //  texels are in this order: (x0,y0) / (x1,y0) / (x0,y1) / (x1,y1)
  float sdf1[4] = {cell_sdf1.d00, cell_sdf1.d10, cell_sdf1.d01, cell_sdf1.d11};
  float sdf2[4] = {cell_sdf2.d00, cell_sdf2.d10, cell_sdf2.d01, cell_sdf2.d11};

  printf("real sdf1:%f,%f,%f,%f\n", sdf1[0], sdf1[1], sdf1[2],
         sdf1[3]); // looks
                   // fine
  printf("real sdf2:%f,%f,%f,%f\n", sdf2[0], sdf2[1], sdf2[2],
         sdf2[3]); // looks
                   // fine

  float written_total_sdf[4]; // values written by the shader
  for (int i = 0; i < 4; i++) {
    uint8_t low = encode_distance_4bits(sdf1[i]);
    uint8_t high = encode_distance_4bits(sdf2[i]);

    uint8_t total = (high << 4) | low;
    written_total_sdf[i] = gpu_int8_to_norm_float(total);
  }

  float point[2] = {0.001f / h, 0.002f / h};

  {
    // sampling based on the real sdf1 and sdf2 distances, not the encoded ones.
    float field1 = sample(point[0], point[1], sdf1);

    float field2 = sample(point[0], point[1], sdf2);

    printf("real_f1:%f, real_f2:%f\n", field1,
           field2); // field1 and field2 look right
  }

  float total_sampled = sample(point[0], point[1], written_total_sdf);
  uint8_t total_sampled_int = gpu_norm_float_to_int8(total_sampled);
  uint8_t low = total_sampled_int & 0xF;
  uint8_t high = (total_sampled_int >> 4) & 0xF;
  float f1_decoded = decode_distance_4bits(low);
  float f2_decoded = decode_distance_4bits(high);

  printf("total:%f, %d, f1_dec:%f, f2_dec:%f \n", total_sampled,
         total_sampled_int, f1_decoded, f2_decoded);

  /*
  float sdf1_4encoded[4];
  float sdf2_4encoded[4];

  for (int i = 0; i < 4; i++) {
    sdf1_4encoded[i] = sdf1[i] / 255.0;
  }

  printf("total:%d, f1:%d, f2:%d\n", total_int, field1_int, field2_int);
  */
  return 0;
}