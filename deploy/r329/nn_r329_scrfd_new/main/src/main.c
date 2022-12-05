#include "stdio.h"
#include <stdint.h>
#include <stdbool.h>

#include "global_config.h"
#include "global_build_info_time.h"
#include "global_build_info_version.h"

#include "libmaix_cam.h"
#include "libmaix_disp.h"
#include "libmaix_image.h"
// #include "libmaix_nn.h"
#include "main.h"
#include <sys/time.h>
#include <unistd.h>
#include <math.h>
#include <signal.h>

#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <getopt.h>
#include <sys/mman.h>
#include <errno.h>
#include "mdsc.h"
#include <string.h>

#include "standard_api.h"
#include "libmaix_cv_image.h"
#include "scrfd_decode_cv.h"

#define LOAD_IMAGE 1
#if LOAD_IMAGE
#define SAVE_NETOUT 0
#endif
#define debug_line printf("%s:%d %s %s %s \r\n", __FILE__, __LINE__, __FUNCTION__, __DATE__, __TIME__)


#define DISPLAY_TIME 1

#if DISPLAY_TIME
    struct timeval start, end;
    int64_t interval_s;
#define CALC_TIME_START()           \
    do                              \
    {                               \
        gettimeofday(&start, NULL); \
    } while (0)
#define CALC_TIME_END(name)                                                               \
    do                                                                                    \
    {                                                                                     \
        gettimeofday(&end, NULL);                                                         \
        interval_s = (int64_t)(end.tv_sec - start.tv_sec) * 1000000ll;                    \
        printf("%s use time: %lld us\n", name, interval_s + end.tv_usec - start.tv_usec); \
    } while (0)
#else
#define CALC_TIME_START()
#define CALC_TIME_END(name)
#endif



static volatile bool program_exit = false;

int loadFromBin(const char *binPath, int size, signed char *buffer)
{
    FILE *fp = fopen(binPath, "rb");
    if (fp == NULL)
    {
        fprintf(stderr, "fopen %s failed\n", binPath);
        return -1;
    }
    int nread = fread(buffer, 1, size, fp);
    if (nread != size)
    {
        fprintf(stderr, "fread bin failed %d\n", nread);
        return -1;
    }
    fclose(fp);

    return 0;
}

int save_bin(const char *path, int size, uint8_t *buffer)
{
    FILE *fp = fopen(path, "wb");
    if (fp == NULL)
    {
        fprintf(stderr, "fopen %s failed\n", path);
        return -1;
    }
    int nwrite = fwrite(buffer, 1, size, fp);
    if (nwrite != size)
    {
        fprintf(stderr, "fwrite bin failed %d\n", nwrite);
        return -1;
    }
    fclose(fp);

    return 0;
}

static void softmax(float *data, int n )
{
    int stride = 1;
    int i;
    // int diff;
    // float e;
    float sum = 0;
    float largest_i = data[0];

    for (i = 0; i < n; ++i)
    {
        if (data[i * stride] > largest_i)
            largest_i = data[i * stride];
    }
    for (i = 0; i < n; ++i)
    {
        float value = expf(data[i * stride] - largest_i);
        sum += value;
        data[i * stride] = value;
    }
    for (i = 0; i < n; ++i)
	{
        data[i * stride] /= sum;
	}
}

int label_oft = 1;

typedef struct {
    int index;
    int8_t val;
} int8_data_t;

typedef struct {
    int index;
    uint8_t val;
} uint8_data_t;

int uint8_comp_down(const void*p1, const void*p2) {
	//<0, 元素1排在元素2之前；即降序
	int tmp = (((uint8_data_t*)p2)->val) - (((uint8_data_t*)p1)->val);
	return tmp;  
}

int int8_comp_down(const void*p1, const void*p2) {
	//<0, 元素1排在元素2之前；即降序
	int tmp = (((int8_data_t*)p2)->val) - (((int8_data_t*)p1)->val);
	return tmp;  
}


static void decode_result_int8(int8_t *result, uint32_t size, int* label_idx, int* prob)
{
    int8_data_t* buf = (int8_data_t*)malloc(sizeof(int8_data_t)*size);
    if(buf == NULL) return;
    for(int i=0; i < size; i++) {
        buf[i].index = i;
        buf[i].val = result[i];
    }
    qsort(buf, size, sizeof(int8_data_t), int8_comp_down);
    printf("Decode Result:\r\n");
    for(int i=0; i < 5; i++) {
        printf("    %d: class %4d, prob %3d; label: %s\r\n", i, buf[i].index, buf[i].val, labels[buf[i].index-label_oft]);
    }
    *label_idx = buf[0].index;
    *prob = buf[0].val;
    free(buf);
    return;
}

static void decode_result_uint8(uint8_t *result, uint32_t size, int* label_idx, int* prob)
{
    uint8_data_t* buf = (uint8_data_t*)malloc(sizeof(uint8_data_t)*size);
    if(buf == NULL) return;
    for(int i=0; i < size; i++) {
        buf[i].index = i;
        buf[i].val = result[i];
    }
    qsort(buf, size, sizeof(uint8_data_t), uint8_comp_down);
    printf("Decode Result:\r\n");
    for(int i=0; i < 5; i++) {
        printf("    %d: class %4d, prob %3u; label: %s\r\n", i, buf[i].index, buf[i].val, labels[buf[i].index-label_oft]);
    }
    *label_idx = buf[0].index;
    *prob = buf[0].val;
    free(buf);
    return;
}

int init_graph(char* file_model, aipu_ctx_handle_t ** ctx, aipu_graph_desc_t* gdesc, aipu_buffer_alloc_info_t* info)
{
    const char* status_msg =NULL;
    aipu_status_t status = AIPU_STATUS_SUCCESS;
    int ret = 0;
    //Step1: init ctx handle
    status = AIPU_init_ctx(ctx);       
    if (status != AIPU_STATUS_SUCCESS) {
        AIPU_get_status_msg(status, &status_msg);
        printf("[DEMO ERROR] AIPU_init_ctx: %s\n", status_msg);
        ret = -1;
        //goto out;
    }

    //Step2: load graph
    status = AIPU_load_graph_helper(*ctx, file_model, gdesc);
    if (status != AIPU_STATUS_SUCCESS) {
        AIPU_get_status_msg(status, &status_msg);
        printf("[DEMO ERROR] AIPU_load_graph_helper: %s\n", status_msg);
        ret = -2;
        //goto deinit_ctx;
    }
    printf("[DEMO INFO] AIPU load graph successfully.\n");

    //Step3: alloc tensor buffers
    status = AIPU_alloc_tensor_buffers(*ctx, gdesc, info);
    if (status != AIPU_STATUS_SUCCESS) {
        AIPU_get_status_msg(status, &status_msg);
        printf("[DEMO ERROR] AIPU_alloc_tensor_buffers: %s\n", status_msg);
        ret = -3;
        //goto unload_graph;
    }
    
    return ret;
}

int infer_img(libmaix_image_t *ai_frame, aipu_ctx_handle_t ** ctx, aipu_graph_desc_t* gdesc, aipu_buffer_alloc_info_t* info, float** score_blobs, float** bbox_blobs, float** kps_blobs)
{
    uint32_t job_id=0;
    const char* status_msg =NULL;
    int32_t time_out=-1;
    bool finish_job_successfully = true;
    aipu_status_t status = AIPU_STATUS_SUCCESS;
    int ret = 0;
    
    memcpy(info->inputs.tensors[0].va, ai_frame->data, info->inputs.tensors[0].size);
    status = AIPU_create_job(*ctx, gdesc, info->handle, &job_id);
    if (status != AIPU_STATUS_SUCCESS) {
        AIPU_get_status_msg(status, &status_msg);
        printf("[DEMO ERROR] AIPU_create_job: %s\n", status_msg);
        ret = -1;
        //goto free_tensor_buffers;
    }
    status = AIPU_finish_job(*ctx, job_id, time_out);
    if (status != AIPU_STATUS_SUCCESS) {
        AIPU_get_status_msg(status, &status_msg);
        printf("[DEMO ERROR] AIPU_finish_job: %s\n", status_msg);
        finish_job_successfully = false;
    } else {
        finish_job_successfully = true;
    }

    if (finish_job_successfully) {
        int8_t *output_score_8_int8 = (int8_t *)info->outputs.tensors[0].va;
        int8_t *output_socre_16_int8 = (int8_t *)info->outputs.tensors[1].va;
        int8_t *output_socre_32_int8 = (int8_t *)info->outputs.tensors[2].va;
        int8_t *output_bbox_8_int8 = (int8_t *)info->outputs.tensors[3].va;
        int8_t *output_bbox_16_int8 = (int8_t *)info->outputs.tensors[4].va;
        int8_t *output_bbox_32_int8 = (int8_t *)info->outputs.tensors[5].va;
        int8_t *output_kps_8_int8 = (int8_t *)info->outputs.tensors[6].va;
        int8_t *output_kps_16_int8 = (int8_t *)info->outputs.tensors[7].va;
        int8_t *output_kps_32_int8 = (int8_t *)info->outputs.tensors[8].va;
        uint32_t size_socre_8 = info->outputs.tensors[0].size;
        uint32_t size_socre_16 = info->outputs.tensors[1].size;
        uint32_t size_socre_32 = info->outputs.tensors[2].size;
        uint32_t size_bbox_8 = info->outputs.tensors[3].size;
        uint32_t size_bbox_16 = info->outputs.tensors[4].size;
        uint32_t size_bbox_32 = info->outputs.tensors[5].size;
        uint32_t size_kps_8 = info->outputs.tensors[6].size;
        uint32_t size_kps_16 = info->outputs.tensors[7].size;
        uint32_t size_kps_32 = info->outputs.tensors[8].size;

        // for (int i = 0; i < size_socre_8; i++) {
        //     score_blobs[0][i] = (float)output_score_8_int8[i] * gdesc->outputs.desc[0].scale;
            
        // }
        // printf("[DEMO INFO] score_blobs[0] size: %d scale: %f zeropoint: %f\n", size_socre_8, gdesc->outputs.desc[0].scale, gdesc->outputs.desc[0].zero_point);
        // printf("[DEMO INFO] score_blobs[1] size: %d scale: %f zeropoint: %f\n", size_socre_16, gdesc->outputs.desc[1].scale, gdesc->outputs.desc[1].zero_point);
        // printf("[DEMO INFO] score_blobs[2] size: %d scale: %f zeropoint: %f\n", size_socre_32, gdesc->outputs.desc[2].scale, gdesc->outputs.desc[2].zero_point);
        // printf("[DEMO INFO] bbox_blobs[0] size: %d scale: %f zeropoint: %f\n", size_bbox_8, gdesc->outputs.desc[3].scale, gdesc->outputs.desc[3].zero_point);
        // printf("[DEMO INFO] bbox_blobs[1] size: %d scale: %f zeropoint: %f\n", size_bbox_16, gdesc->outputs.desc[4].scale, gdesc->outputs.desc[4].zero_point);
        // printf("[DEMO INFO] bbox_blobs[2] size: %d scale: %f zeropoint: %f\n", size_bbox_32, gdesc->outputs.desc[5].scale, gdesc->outputs.desc[5].zero_point);
        // printf("[DEMO INFO] kps_blobs[0] size: %d scale: %f zeropoint: %f\n", size_kps_8, gdesc->outputs.desc[6].scale, gdesc->outputs.desc[6].zero_point);
        // printf("[DEMO INFO] kps_blobs[1] size: %d scale: %f zeropoint: %f\n", size_kps_16, gdesc->outputs.desc[7].scale, gdesc->outputs.desc[7].zero_point);
        // printf("[DEMO INFO] kps_blobs[2] size: %d scale: %f zeropoint: %f\n", size_kps_32, gdesc->outputs.desc[8].scale, gdesc->outputs.desc[8].zero_point);

        for (int i = 0; i < size_socre_8; i++) {
            score_blobs[0][i] = (float)output_score_8_int8[i] / gdesc->outputs.desc[0].scale;
        }
        for (int i = 0; i < size_socre_16; i++) {
            score_blobs[1][i] = (float)output_socre_16_int8[i] / gdesc->outputs.desc[1].scale;
        }
        for (int i = 0; i < size_socre_32; i++) {
            score_blobs[2][i] = (float)output_socre_32_int8[i] / gdesc->outputs.desc[2].scale;
        }
        for (int i = 0; i < size_bbox_8; i++) {
            bbox_blobs[0][i] = (float)output_bbox_8_int8[i] / gdesc->outputs.desc[3].scale;
        }
        for (int i = 0; i < size_bbox_16; i++) {
            bbox_blobs[1][i] = (float)output_bbox_16_int8[i] / gdesc->outputs.desc[4].scale;
        }
        for (int i = 0; i < size_bbox_32; i++) {
            bbox_blobs[2][i] = (float)output_bbox_32_int8[i] / gdesc->outputs.desc[5].scale;
        }
        for (int i = 0; i < size_kps_8; i++) {
            kps_blobs[0][i] = (float)output_kps_8_int8[i] / gdesc->outputs.desc[6].scale;
        }
        for (int i = 0; i < size_kps_16; i++) {
            kps_blobs[1][i] = (float)output_kps_16_int8[i] / gdesc->outputs.desc[7].scale;
        }
        for (int i = 0; i < size_kps_32; i++) {
            kps_blobs[2][i] = (float)output_kps_32_int8[i] / gdesc->outputs.desc[8].scale;
        }

        //sigmoid score
        for (int i = 0; i < size_socre_8; i++) {
            score_blobs[0][i] = 1.0 / (1.0 + exp(-score_blobs[0][i]));
        }
        for (int i = 0; i < size_socre_16; i++) {
            score_blobs[1][i] = 1.0 / (1.0 + exp(-score_blobs[1][i]));
        }
        for (int i = 0; i < size_socre_32; i++) {
            score_blobs[2][i] = 1.0 / (1.0 + exp(-score_blobs[2][i]));
        }
    }

    status = AIPU_clean_job(*ctx, job_id);
    if (status != AIPU_STATUS_SUCCESS) {
        AIPU_get_status_msg(status, &status_msg);
        printf("[TEST ERROR] AIPU_clean_job: %s\n", status_msg);
        ret = -2;
        //goto free_tensor_buffers;
    }
    return ret;
}


float cal_fps(struct timeval start, struct timeval end)
{
    struct timeval interval;
    if (end.tv_usec >= start.tv_usec) {
        interval.tv_usec = end.tv_usec - start.tv_usec;
        interval.tv_sec = end.tv_sec - start.tv_sec;
    } else  {
        interval.tv_usec = 1000000 + end.tv_usec - start.tv_usec;
        interval.tv_sec = end.tv_sec - 1 - start.tv_sec;
    }
    float fps = 1000000.0 / interval.tv_usec;
    return fps;
}



void nn_test(struct libmaix_disp *disp)
{

    printf("--image module init\n");
    libmaix_image_module_init();
    // libmaix_nn_module_init();
    libmaix_camera_module_init();
    char *model_path = "/root/aipu_scrfd_res_640_640.bin";
    // float mean[3] = {127.5, 127.5, 127.5};
    // float norm[2] = {-1.0, 1.0};
    aipu_ctx_handle_t *ctx = NULL;
    aipu_status_t status = AIPU_STATUS_SUCCESS;
    const char* status_msg =NULL;
    aipu_graph_desc_t gdesc;
    aipu_buffer_alloc_info_t info;
    int ret = 0;
    ret = init_graph(model_path, &ctx, &gdesc, &info);
    if (ret == -1) {
        goto out;
    }else if (ret == -2) {
        goto deinit_ctx;
    }else if (ret == -3) {
        goto unload_graph;
    }
    // int label_idx, label_prob;
    struct timeval start, end;
    int signed_flag = 1;

    // char *mdsc_path = "/root/mdsc/r329_mobilenet2.mdsc";
    
    uint32_t res_w = 640;
    uint32_t res_h = 640;
    uint32_t cam_w = 640;
    uint32_t cam_h = 480;
    uint32_t lcd_w = disp->width;
    uint32_t lcd_h = disp->height;
    libmaix_nn_t *nn = NULL;
    float* result = NULL;
    libmaix_err_t err = LIBMAIX_ERR_NONE;

    scrfd_object_t *result_objs = NULL;
    int result_num = 0;

    // creat feat_map buffer
    float *output_score_8_data = (float*)malloc(sizeof(float) * 12800);
    float *output_score_16_data = (float*)malloc(sizeof(float) * 3200);
    float *output_score_32_data = (float*)malloc(sizeof(float) * 800);
    float *output_bbox_8_data = (float*)malloc(sizeof(float) * 12800 * 4);
    float *output_bbox_16_data = (float*)malloc(sizeof(float) * 3200 * 4);
    float *output_bbox_32_data = (float*)malloc(sizeof(float) * 800 * 4);
    float *output_kps_8_data =  (float*)malloc(sizeof(float) * 12800 * 8);
    float *output_kps_16_data = (float*)malloc(sizeof(float) * 3200 * 8);
    float *output_kps_32_data = (float*)malloc(sizeof(float) * 800 * 8);

    float* score_blobs[3] = {output_score_8_data, output_score_16_data, output_score_32_data};
    float* bbox_blobs[3] = {output_bbox_8_data, output_bbox_16_data, output_bbox_32_data};
    float* kps_blobs[3] = {output_kps_8_data, output_kps_16_data, output_kps_32_data};

    printf("--cam create\n");
    libmaix_image_t *cam_img = libmaix_image_create(cam_w, cam_h, LIBMAIX_IMAGE_MODE_RGB888, LIBMAIX_IMAGE_LAYOUT_HWC, NULL, true);
    libmaix_image_t *img = libmaix_image_create(res_w, res_h, LIBMAIX_IMAGE_MODE_RGB888, LIBMAIX_IMAGE_LAYOUT_HWC, NULL, true);
    libmaix_image_t *show = libmaix_image_create(disp->width, disp->height, LIBMAIX_IMAGE_MODE_RGB888, LIBMAIX_IMAGE_LAYOUT_HWC, NULL, true);
    libmaix_cam_t *cam = libmaix_cam_create(0, cam_w, cam_h, 1, 1);

    if (!cam)
    {
        printf("create cam fail\n");
    }
    printf("--cam start capture\n");
    err = cam->start_capture(cam);

    if (err != LIBMAIX_ERR_NONE)
    {
        printf("start capture fail: %s\n", libmaix_get_err_msg(err));
        goto end;
    }

    // // input
    // libmaix_nn_layer_t input = {
    //     .w = 224,
    //     .h = 224,
    //     .c = 3,
    //     .dtype = LIBMAIX_NN_DTYPE_INT8,
    //     .data = NULL,
    //     .need_quantization = true,
    //     .buff_quantization = NULL
    // };
    // //output
    // libmaix_nn_layer_t out_fmap = {
    //     .w = 1,
    //     .h = 1,
    //     .c = 1000,
    //     .layout = LIBMAIX_IMAGE_LAYOUT_CHW,
    //     .dtype = LIBMAIX_NN_DTYPE_FLOAT,
    //     .data = NULL
    // };
    // //input buffer
    // int8_t *quantize_buffer = (int8_t *)malloc(input.w * input.h * input.c);
    // if (!quantize_buffer)
    // {
    //     printf("no memory!!!\n");
    //     goto end;
    // }
    // input.buff_quantization = quantize_buffer;
    // // output buffer
    // float *output_buffer = (float *)malloc(out_fmap.c * out_fmap.w * out_fmap.h * sizeof(float));
    // if (!output_buffer)
    // {
    //     printf("no memory!!!\n");
    //     goto end;
    // }
    // out_fmap.data = output_buffer;

    // nn = load_mdsc(mdsc_path);
    printf("-- start loop\n");
    while (!program_exit)
    {
#if LOAD_IMAGE
        printf("-- load input bin file\n");
        // loadFromBin("/root/input.bin", res_w * res_h * 3, img->data);
        libmaix_cv_image_open_file(&img, "/root/hotgen_covid-19.jpg");
        uint8_t mean[3] = {127, 127, 127};
        libmaix_cv_image_substract_mean_normalize(&img, mean, NULL);
#else
        gettimeofday(&start, NULL);
        err = cam->capture_image(cam, &cam_img);

        if (err != LIBMAIX_ERR_NONE)
        {
            // not ready， sleep to release CPU
            if (err == LIBMAIX_ERR_NOT_READY)
            {
                usleep(20 * 1000);
                continue;
            }
            else
            {
                printf("capture fail: %s\n", libmaix_get_err_msg(err));
                break;
            }
        }
        libmaix_cv_image_crop(cam_img, (cam_w - res_w) / 2, (cam_h - res_h) / 2, res_w, res_h, &img);
        libmaix_cv_image_substract_mean_normalize(&img, mean, NULL);

#endif

        // forward process
        // input.data = (uint8_t *)img->data;
        CALC_TIME_START();
        // err = nn->forward(nn, &input, &out_fmap);
        err = infer_img(img, &ctx, &gdesc, &info, &score_blobs, &bbox_blobs, &kps_blobs);
        CALC_TIME_END("forward");
        if (err != 0)
        {
            printf("forward fail!\n");
            goto free_tensor_buffers;
        }

        CALC_TIME_START();
        // printf("%.2f::%s \n", label_prob / 255., labels[label_idx - label_oft]);
        // printf("____________\n");
        //decode
        scrfd_decode(score_blobs, bbox_blobs, kps_blobs, &result_objs, &result_num);


        CALC_TIME_END("decode");
        // err = libmaix_cv_image_resize(img, disp->width, disp->height, &show);
        // draw result
        for (int i = 0; i < result_num; i++)
        {
            libmaix_cv_image_draw_rectangle(img, result_objs[i].bbox.x, result_objs[i].bbox.y, result_objs[i].bbox.x + result_objs[i].bbox.w, result_objs[i].bbox.y + result_objs[i].bbox.h, MaixColor(255, 0, 0), 5);
            // libmaix_cv_image_draw_string(show, result_objs[i].bbox.x, result_objs[i].bbox.y, result_objs[i].label, 1, MaixColor(255, 0, 0), 1);
            // libmaix_cv_image_draw_string(show, result_objs[i].bbox.x, result_objs[i].bbox.y + 16, "score: " + result_objs[i].score, 1, MaixColor(255, 0, 0), 1);
            for (int j = 0; j < MAX_KPS_NUM; j++)
            {
                libmaix_cv_image_draw_circle(img, result_objs[i].kps[j][0], result_objs[i].kps[j][1], 2, MaixColor(0, 255, 0), 5);
            }
        }
#if LOAD_IMAGE
        libmaix_cv_image_resize(img, disp->width, disp->height, &show);
        // print result
        printf("-- show result\n");
        printf("result_num: %d\n", result_num);
        for (int i = 0; i < result_num; i++)
        {
            printf("result[%d]:\n", i);
            printf("\tbbox: %f, %f, %f, %f\n", result_objs[i].bbox.x, result_objs[i].bbox.y, result_objs[i].bbox.x + result_objs[i].bbox.w, result_objs[i].bbox.y + result_objs[i].bbox.h);
            printf("\tscore: %f\n", result_objs[i].score);
            for (int j = 0; j < MAX_KPS_NUM; j++)
            {
                printf("\tkps[%d]: %f, %f\n", j, result_objs[i].kps[j][0], result_objs[i].kps[j][1]);
            }
        }
#if SAVE_NETOUT
        save_bin("/root/output_score_8.bin", 12800 * sizeof(float), output_score_8_data);
        save_bin("/root/output_score_16.bin", 3200 * sizeof(float), output_score_16_data);
        save_bin("/root/output_score_32.bin", 800 * sizeof(float), output_score_32_data);
        save_bin("/root/output_bbox_8.bin", 51200 * sizeof(float), output_bbox_8_data);
        save_bin("/root/output_bbox_16.bin", 12800 * sizeof(float), output_bbox_16_data);
        save_bin("/root/output_bbox_32.bin", 3200 * sizeof(float), output_bbox_32_data);
        save_bin("/root/output_kps_8.bin", 102400 * sizeof(float), output_kps_8_data);
        save_bin("/root/output_kps_16.bin", 25600 * sizeof(float), output_kps_16_data);
        save_bin("/root/output_kps_32.bin", 6400 * sizeof(float), output_kps_32_data);
        printf("-- save output bin file\n");
#endif
#else
        libmaix_cv_image_crop(cam_img, (cam_w - lcd_w) / 2, (cam_h - lcd_h) / 2, lcd_w, lcd_h, &show);
#endif     
        // libmaix_cv_image_draw_string(show, 0, lcd_h - 16, labels[label_idx - label_oft], 1, MaixColor(255,0,0), 1);

        gettimeofday(&end, NULL);
        float fps = cal_fps(start, end);
        char fps_str[16];
        sprintf(fps_str, "%.1ffps", fps);
        libmaix_cv_image_draw_string(show, 0, 0, fps_str, 2, MaixColor(255, 0, 0), 2);
        disp->draw_image(disp, show);
        // clear result
        free(result_objs);
#if LOAD_IMAGE
        break;
#endif
    }

free_tensor_buffers:
    status = AIPU_free_tensor_buffers(ctx, info.handle);
    if (status != AIPU_STATUS_SUCCESS) {
        AIPU_get_status_msg(status, &status_msg);
        printf("[DEMO ERROR] AIPU_free_tensor_buffers: %s\n", status_msg);
        ret = -1;
    }
unload_graph:
    status = AIPU_unload_graph(ctx, &gdesc);
    if (status != AIPU_STATUS_SUCCESS) {
        AIPU_get_status_msg(status, &status_msg);
        printf("[DEMO ERROR] AIPU_unload_graph: %s\n", status_msg);
        ret = -1;
    }
deinit_ctx:
    status = AIPU_deinit_ctx(ctx);
    if (status != AIPU_STATUS_SUCCESS) {
        AIPU_get_status_msg(status, &status_msg);
        printf("[DEMO ERROR] AIPU_deinit_ctx: %s\n", status_msg);
        ret = -1;
    }

out:
end:
    // if (output_buffer)
    // {
    //     free(output_buffer);
    // }
    // if (nn)
    // {
    //     libmaix_nn_destroy(&nn);
    // }
    if (output_score_8_data)
    {
        free(output_score_8_data);
    }
    if (output_bbox_8_data)
    {
        free(output_bbox_8_data);
    }
    if (output_kps_8_data)
    {
        free(output_kps_8_data);
    }
    if (output_score_16_data)
    {
        free(output_score_16_data);
    }
    if (output_bbox_16_data)
    {
        free(output_bbox_16_data);
    }
    if (output_kps_16_data)
    {
        free(output_kps_16_data);
    }
    if (output_score_32_data)
    {
        free(output_score_32_data);
    }
    if (output_bbox_32_data)
    {
        free(output_bbox_32_data);
    }
    if (output_kps_32_data)
    {
        free(output_kps_32_data);
    }


    if (cam)
    {
        printf("--cam destory\n");
        libmaix_cam_destroy(&cam);
    }
    printf("--image module deinit\n");
    // libmaix_nn_module_deinit();
    libmaix_image_module_deinit();
}

static void handle_signal(int signo)
{
    if (SIGINT == signo || SIGTSTP == signo || SIGTERM == signo || SIGQUIT == signo || SIGPIPE == signo || SIGKILL == signo)
    {
        program_exit = true;
    }
}

int main(int argc, char *argv[])
{
    struct libmaix_disp *disp = libmaix_disp_create(0);
    if (disp == NULL)
    {
        printf("creat disp object fail\n");
        return -1;
    }

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    printf("program start\n");
    nn_test(disp);
    printf("program end\n");

    libmaix_disp_destroy(&disp);
    return 0;
}
