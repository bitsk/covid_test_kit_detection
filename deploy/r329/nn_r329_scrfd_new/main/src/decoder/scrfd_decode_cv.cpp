
#include <vector>
#include <iostream>
#include <map>
#include <cmath>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include "scrfd_decode_cv.h"


const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
const float scale_vals[3] = {0.078125f, 0.078125f, 0.078125f};
unsigned int fmc = 3; // feature map count
bool use_kps = true;
unsigned int num_anchors = 2;
std::vector<int> feat_stride_fpn = {8, 16, 32};
std::unordered_map<int, std::vector<scrfd_center_point_t>> center_points;
bool center_point_is_update = false;
int nms_pre = 100;
int max_nms = 3000;

int input_width = 640;
int input_height = 640;

float score_threshold = 0.5f;
float nms_threshold = 0.45f;

scrfd_scale_params scale_params = {1.0, 0, 0, 1};

void scrfd_generate_points(const int target_height, const int target_width)
{
    if (center_point_is_update)
    {
        return;
    }
    for (auto stride: feat_stride_fpn)
    {
        unsigned int num_grid_w = target_width / stride;
        unsigned int num_grid_h = target_height / stride;

        for (unsigned int i = 0; i < num_grid_h; i++)
        {
            for (unsigned int j = 0; j < num_grid_w; j++)
            {
                for (unsigned int k = 0; k < num_anchors; k++)
                {
                    scrfd_center_point_t center_point;
                    center_point.cx = (float)j;
                    center_point.cy = (float)i;
                    center_point.stride = (float)stride;
                    center_points[stride].push_back(center_point);
                }
            }
        }
    }
    center_point_is_update = true;
}

void scrfd_generate_bboxes_kps_single_stride(const scrfd_scale_params &scale_params,
                                             float* score_pred, float* bbox_pred, float* kps_pred,
                                             unsigned int stride, float score_threshold, float img_height, float img_width, std::vector<scrfd_object_t> &objects_collection)
{
    unsigned int nms_pre_ = (stride / 8) * nms_pre;
    nms_pre_ = nms_pre_ >= nms_pre ? nms_pre_ : nms_pre;

    int blob_width = input_width / stride;
    int blob_height = input_height / stride;

    const unsigned int num_points = blob_width * blob_height * num_anchors;

    const float* score_ptr = score_pred;
    const float* bbox_ptr = bbox_pred;
    const float* kps_ptr = kps_pred;

    float ratio = scale_params.ratio;
    int dw = scale_params.dw;
    int dh = scale_params.dh;

    unsigned int count = 0;
    auto &stride_points = center_points[stride];

    // printf("objects_collection.size() = %d\n", objects_collection.size());

    for (unsigned int i = 0; i < num_points; i++)
    {
        const float cls_conf = score_ptr[i];
        if (cls_conf < score_threshold)
        {
            continue;
        }
        auto &point = stride_points.at(i);
        const float cx = point.cx;
        const float cy = point.cy;
        const float s = point.stride;

        const float *offsets = bbox_ptr + i * 4;
        float l = offsets[0];
        float t = offsets[1];
        float r = offsets[2];
        float b = offsets[3];

        scrfd_object_t object;
        float x1 = ((cx - l) * s - (float) dw) / ratio;
        float y1 = ((cy - t) * s - (float) dh) / ratio;
        float x2 = ((cx + r) * s - (float) dw) / ratio;
        float y2 = ((cy + b) * s - (float) dh) / ratio;

        x1 = std::max(0.f, x1);
        y1 = std::max(0.f, y1);
        x2 = std::min(img_width - 1.f, x2);
        y2 = std::min(img_height - 1.f, y2);

        object.bbox.x = x1;
        object.bbox.y = y1;
        object.bbox.w = x2 - x1;
        object.bbox.h = y2 - y1;
        object.score = cls_conf;

        const float *kps_offsets = kps_ptr + i * 2 * MAX_KPS_NUM;
        for (int j = 0; j < MAX_KPS_NUM; j++)
        {
            float kps_l = kps_offsets[j * 2];
            float kps_t = kps_offsets[j * 2 + 1];
            float kps_x = ((cx + kps_l) * s - (float) dw) / ratio;
            float kps_y = ((cy + kps_t) * s - (float) dh) / ratio;
            kps_x = std::min(std::max(0.f, kps_x), img_width - 1.f);
            kps_y = std::min(std::max(0.f, kps_y), img_height - 1.f);
            object.kps[j][0] = kps_x;
            object.kps[j][1] = kps_y;
        }

        objects_collection.push_back(object);
        count += 1;
        if (count >= nms_pre_)
        {
            break;
        }
    }

    // printf("objects_collection.size() = %d\n", objects_collection.size());

    if (objects_collection.size() > nms_pre_) {
        std::sort(
                objects_collection.begin(),
                objects_collection.end(),
                [](const scrfd_object_t &a, const scrfd_object_t &b) {
                    return a.score > b.score;
                }
        );
        objects_collection.resize(nms_pre_);
    }
}

static float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

static float intersection_area(const scrfd_object_t& a, const scrfd_object_t& b)
{
    float w = overlap(a.bbox.x, a.bbox.w, b.bbox.x, b.bbox.w);
    float h = overlap(a.bbox.y, a.bbox.h, b.bbox.y, b.bbox.h);
    if (w < 0 || h < 0)
        return 0;
    return w * h;
}

static float union_area(const scrfd_object_t& a, const scrfd_object_t& b)
{
    return a.bbox.w * a.bbox.h + b.bbox.w * b.bbox.h - intersection_area(a, b);
}

static float box_iou(const scrfd_object_t& a, const scrfd_object_t& b)
{
    return intersection_area(a, b) / union_area(a, b);
}

void scrfd_nms_bboxes_kps(std::vector<scrfd_object_t> &input, std::vector<scrfd_object_t> &output, float nms_threshold, unsigned int topk)
{
    if (input.empty())
    {
        return;
    }
    std::sort(
            input.begin(),
            input.end(),
            [](const scrfd_object_t &a, const scrfd_object_t &b) {
                return a.score > b.score;
            }
    );
    const unsigned int box_num = input.size();
    std::vector<int> merged(box_num, 0);

    unsigned int count = 0;
    for (unsigned int i = 0; i < box_num; i++)
    {
        if (merged[i])
        {
            continue;
        }
        std::vector<scrfd_object_t> buf;
        buf.push_back(input[i]);
        merged[i] = 1;
        for (unsigned int j = i + 1; j < box_num; j++)
        {
            if (merged[j])
            {
                continue;
            }
            float iou = box_iou(input[i], input[j]);
            if (iou > nms_threshold)
            {
                buf.push_back(input[j]);
                merged[j] = 1;
            }
        }
        output.push_back(buf[0]);
        count += 1;
        if (count >= topk)
        {
            break;
        }
    }
}

void scrfd_generate_objects(std::vector<scrfd_object_t> &objects_collection,
                            std::vector<float*> &feat_maps)
{
    scrfd_generate_points(input_height, input_width);
    objects_collection.clear();
    // printf("objects_collection.size() = %d\n", objects_collection.size());
    std::vector<scrfd_object_t> objects_collection_tmp;

    float *score_8 = feat_maps.at(0);
    float *score_16 = feat_maps.at(1);
    float *score_32 = feat_maps.at(2);
    float *bbox_8 = feat_maps.at(3);
    float *bbox_16 = feat_maps.at(4);
    float *bbox_32 = feat_maps.at(5);
    if (use_kps)
    {
        float *kps_8 = feat_maps.at(6);
        float *kps_16 = feat_maps.at(7);
        float *kps_32 = feat_maps.at(8);
        
        scrfd_generate_bboxes_kps_single_stride(scale_params, score_8, bbox_8, kps_8, 8, score_threshold, input_height, input_width, objects_collection_tmp);
        scrfd_generate_bboxes_kps_single_stride(scale_params, score_16, bbox_16, kps_16, 16, score_threshold, input_height, input_width, objects_collection_tmp);
        scrfd_generate_bboxes_kps_single_stride(scale_params, score_32, bbox_32, kps_32, 32, score_threshold, input_height, input_width, objects_collection_tmp);
    }

    scrfd_nms_bboxes_kps(objects_collection_tmp, objects_collection, nms_threshold, 100);
}

int scrfd_decode(float **score_blobs, float **bbox_blobs, float **kps_blobs, scrfd_object_t **objects, int *num_objects)
{
    std::vector<float*> feat_maps;
    for (int i = 0; i < fmc; i++)
    {
        feat_maps.push_back(score_blobs[i]);
    }
    for (int i = 0; i < fmc; i++)
    {
        feat_maps.push_back(bbox_blobs[i]);
    }
    for (int i = 0; i < fmc; i++)
    {
        feat_maps.push_back(kps_blobs[i]);
    }

    std::vector<scrfd_object_t> objects_collection;
    scrfd_generate_objects(objects_collection, feat_maps);

    *num_objects = objects_collection.size();
    // *objects = new scrfd_object_t[*num_objects];
    *objects = (scrfd_object_t*)malloc(sizeof(scrfd_object_t) * *num_objects);
    for (int i = 0; i < *num_objects; i++)
    {
        (*objects)[i] = objects_collection[i];
    }

    return 0;
}