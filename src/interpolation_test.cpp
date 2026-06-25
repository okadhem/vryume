#include <stdint.h>
#include <math.h>
#include <stdio.h>

#include <math.h>
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
uint8_t encode_distance(float distance) {
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
// fx, fy are between 0 and 1
// texels are in this order: (x0,y0) / (x1,y0) / (x0,y1) / (x1,y1)
float sample(float fx, float fy, float texels[4]) {
  // v=v00​(1−fx​)(1−fy​)+v10​fx​(1−fy​)+v01​(1−fx​)fy​+v11​fx​fy
  return texels[0] * (1 - fx) * (1 - fy) + texels[1] * fx * (1 - fy) +
         texels[2] * (1 - fx) * fy + texels[3] * fx * fy;
}

int main() {
  vec2 cell = {0.0f, 0.0f};
  float h = 1.0f;
  vec2 linePoint1 = {0.0f, 0.0f};
  vec2 lineDir1 = {1.0f, 1.0f};
  SdfCell cell_sdf1 = computeCellSDF(cell, h, linePoint1, lineDir1);

  vec2 linePoint2 = {0.5f, 0.0f};
  vec2 lineDir2 = {0.0f, 1.0f};
  SdfCell cell_sdf2 = computeCellSDF(cell, h, linePoint2, lineDir2);

  // p00 ---- p10
  //  |        |
  // p01 ---- p11
  //  texels are in this order: (x0,y0) / (x1,y0) / (x0,y1) / (x1,y1)
  float sdf1[4] = {cell_sdf1.d00, cell_sdf1.d10, cell_sdf1.d01, cell_sdf1.d11};
  float sdf2[4] = {cell_sdf2.d00, cell_sdf2.d10, cell_sdf2.d01, cell_sdf2.d11};

  float written_total_sdf[4]; // values written by the shader
  for (int i = 0; i < 4; i++) {
    uint8_t low = encode_distance(sdf1[i]);
    uint8_t high = encode_distance(sdf2[i]);

    uint8_t total = (high << 4) | low;
    written_total_sdf[i] = ((float)total) / 255.0;
  }

  float point[2] = {0.25, 0.5};

  float total = sample(point[0], point[1], written_total_sdf);
  uint8_t total_int = (uint8_t)lroundf(total * 255.0f);

  float field1 = sample(point[0], point[1], sdf1);
  uint8_t field1_int = (uint8_t)lroundf(field1 * 255.0f);

  float field2 = sample(point[0], point[1], sdf2);
  uint8_t field2_int = (uint8_t)lroundf(field2 * 255.0f);

  printf("total:%d, f1:%d, f2:%d", total_int, field1_int, field2_int);
  return 0;
}