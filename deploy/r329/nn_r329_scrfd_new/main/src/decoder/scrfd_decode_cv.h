#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    float cx;
    float cy;
    float stride;    
}scrfd_center_point_t;

typedef struct
{
    float ratio;
    int dw;
    int dh;
    int flag;
}scrfd_scale_params;

typedef struct 
{
    float x;
    float y;
    float w;
    float h;
}scrfd_box_t;

#define MAX_KPS_NUM 4

typedef struct
{
    scrfd_box_t bbox;
    float score;
    float kps[MAX_KPS_NUM][2];
}scrfd_object_t;

int scrfd_decode(float **score_blobs, float **bbox_blobs, float **kps_blobs, scrfd_object_t **objects, int *num_objects);

#ifdef __cplusplus
}
#endif