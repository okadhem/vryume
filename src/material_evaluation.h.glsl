
//TODO: TWO_SURFACE_VOXEL_2 is now used for things unrealted to two surface voxels, like a marker
// for null voxel info, update the comment bellow.

// Each voxel material byte is made of a header and a payload:
// header (2 bits): ONE_SURFACE_VOXEL, TWO_SURFACE_VOXEL_1, TWO_SURFACE_VOXEL_2, THREE_SURFACE_VOXEL
// ONE_SURFACE_VOXEL, payload(6 bits) = material_id
// in the case of two surfaces on a voxel, we have 3 different configurations:
// - C12: surfaces are on channel 1 and 2
// - C13: surfaces are on channel 1 and 3
// - C23: surfaces are on channel 2 and 3
// one bit of payload will be used to split TWO_SURFACE_VOXEL_1 and TWO_SURFACE_VOXEL_2 in to two sub spaces
// TWO_SURFACE_VOXEL_1 with config bit = 0 => configuration C12
// TWO_SURFACE_VOXEL_1 with config bit = 1 => configuration C13
// TWO_SURFACE_VOXEL_2 with config bit = 0 => configuration C23
// TWO_SURFACE_VOXEL_2 with config bit = 1 => unused
// in all 3 cases, the remaining 5 bits will be the index of the neighbour with the lower channel number,
// i.e for configuration C12 the neighbour index of the surface mapped by channel 1.
// THREE_SURFACE_VOXEL, nothing in payload

const uint HEADER_MASK = 0xC0u; // 0b11000000u;
const uint HEADER_TWO_SURFACE_CONFIG_MASK = 0xE0u; // 0b11100000u;
const uint ONE_SURFACE_VOXEL_HEADER = 0u; // 0b00000000u;
const uint TWO_SURFACE_VOXEL_1_HEADER = 0x40u; // 0b01000000u;
const uint TWO_SURFACE_VOXEL_C12_CONFIG = 0x40u; // 0b01000000u;
const uint TWO_SURFACE_VOXEL_C13_CONFIG = 0x60u; //0b01100000u;
// TWO_SURFACE_VOXEL_2_HEADER no longer to be used
//const uint TWO_SURFACE_VOXEL_2_HEADER = 0x80u; //0b10000000u;
const uint TWO_SURFACE_VOXEL_C23_CONFIG = 0x80u; //0b10000000u;
const uint THREE_SURFACE_VOXEL_HEADER = 0xC0u; //0b11000000u;

//voxel value for the uninitialized state of a voxel.
const uint NULL_VOXEL = 0xA0; //0b10100000u

// converts between coordinates centred around a voxel, ranging from -1 to 1 and linearized id.
uint coord_to_neighbour_id(ivec3 c) {
    // coordinates relative to the lower corner "neighbour" in the 3x3 block.
    uint x_corner = c.x + 1;
    uint y_corner = c.y + 1;
    uint z_corner = c.z + 1;
    return x_corner + 3 * y_corner + 9 * z_corner;
}
// converts between coordinates centred around a voxel, ranging from -1 to 1 and linearized id.
ivec3 neighbour_id_to_coord(uint neighbour_id) {
    uint x_corner = neighbour_id % 3;
    uint y_corner = (neighbour_id / 3) % 3;
    uint z_corner = neighbour_id / 9;
    return ivec3(x_corner, y_corner, z_corner) + ivec3(-1, -1, -1);
}

// returns material ids of voxel v, in ascending channel order. (example, material_id of channel 2 then channel 3)
uint[2] read_2_surface_voxel_material_id(ivec3 v) {
    //TODO we should accept this texel as input
    uint texel = imageLoad(tile_voxel_material, v).r;
    uint low_channel_neighbour_id = texel & ~HEADER_TWO_SURFACE_CONFIG_MASK;
    ivec3 low_channel_voxel = neighbour_id_to_coord(low_channel_neighbour_id) + v;
    uint low_neighbour_texel = imageLoad(tile_voxel_material, low_channel_voxel).r;
    uint low_channel_material_id = low_neighbour_texel & ~HEADER_MASK;

    for (int k = -1; k <= 1; ++k)
        for (int j = -1; j <= 1; ++j)
            for (int i = -1; i <= 1; ++i) {
                ivec3 offset = ivec3(i, j, k);
                if (offset == ivec3(0))
                    continue;
                uint neighbour_texel = imageLoad(tile_voxel_material, v + offset).r;

                if ((neighbour_texel & HEADER_MASK) == ONE_SURFACE_VOXEL_HEADER) {
                    uint neighbour_material_id = neighbour_texel & ~HEADER_MASK;
                    if (neighbour_material_id != low_channel_material_id) {
                        return uint[2](low_channel_material_id, neighbour_material_id);
                    }
                }
            }
    // this should not happen, we must find two material id in the neighbours.
    // TODO report an error.
}

// returns material ids of voxel v, in ascending channel order.
uint[3] read_3_surface_voxel_material_id(ivec3 v) {
    // there might me a better way, if can collect material ids from 1 surface voxels in the neighbourhood
    // then sort them. i am not 100% sure that we will find all of them in the immediate neighbours
    // and we can't go too far without risk.

    uint[3] result;
    for (int k = -1; k <= 1; ++k)
        for (int j = -1; j <= 1; ++j)
            for (int i = -1; i <= 1; ++i) {
                ivec3 offset = ivec3(i, j, k);
                if (offset == ivec3(0))
                    continue;
                uint neighbour_texel = imageLoad(tile_voxel_material, v + offset).r;

                // we are a 3 surface voxels, all our 2 surface voxels neighbours agree with us on channel mapping.

                if ((neighbour_texel & HEADER_TWO_SURFACE_CONFIG_MASK) == TWO_SURFACE_VOXEL_C12_CONFIG) {
                    uint[2] materials = read_2_surface_voxel_material_id(v);
                    result[0] = materials[0];
                    result[1] = materials[1];
                }

                if ((neighbour_texel & HEADER_TWO_SURFACE_CONFIG_MASK) == TWO_SURFACE_VOXEL_C13_CONFIG) {
                    uint[2] materials = read_2_surface_voxel_material_id(v);
                    result[0] = materials[0];
                    result[2] = materials[1];
                }

                if ((neighbour_texel & HEADER_TWO_SURFACE_CONFIG_MASK) == TWO_SURFACE_VOXEL_C23_CONFIG) {
                    uint[2] materials = read_2_surface_voxel_material_id(v);
                    result[1] = materials[0];
                    result[2] = materials[1];
                }
            }
    return result;
}
