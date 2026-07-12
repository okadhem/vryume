struct Edit {
    bool is_removal;
    int material_id;
    uint primitive_type;
    // manual scalar array until we activate standard layout on unifrom buffers
    float param0;
    float param1;
    float param2;
    float param3;

    // transform in model space
    vec4 rotation; // unit quaternion
    vec3 translation;
};

const uint EDIT_PRIMITIVE_SPHERE = 1;
const uint EDIT_PRIMITIVE_BOX = 2;
