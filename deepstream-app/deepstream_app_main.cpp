/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "deepstream_app.h"
#include "deepstream_config_file_parser.h"
#include <cuda_runtime_api.h>
#include "nvds_version.h"
#include "nvdsmeta_schema.h"

#include <iostream>
#include <sstream>
#include "nvds_analytics_meta.h"

#include <string.h>
#include <unistd.h>
#include <termios.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <json-glib/json-glib.h>

#define MAX_INSTANCES 128
#define APP_TITLE "DeepStream"

#define DEFAULT_X_WINDOW_WIDTH 1920
#define DEFAULT_X_WINDOW_HEIGHT 1080

#define MAX_TIME_STAMP_LEN (64)
#define MAX_COUNT_LEN (16)

#ifdef EN_DEBUG
#define LOGD(...) printf(__VA_ARGS__)
#else
#define LOGD(...)
#endif

AppCtx *appCtx[MAX_INSTANCES];
static guint cintr = FALSE;
static GMainLoop *main_loop = NULL;
static gchar **cfg_files = NULL;
static gchar **input_uris = NULL;
static gboolean print_version = FALSE;
static gboolean show_bbox_text = FALSE;
static gboolean print_dependencies_version = FALSE;
static gboolean quit = FALSE;
static gint return_value = 0;
static guint num_instances;
static guint num_input_uris;
static GMutex fps_lock;
static gdouble fps[MAX_SOURCE_BINS];
static gdouble fps_avg[MAX_SOURCE_BINS];

static guint count_offset[10] = { 0 };
static guint count_current[10] = { 0 };

static Display *display = NULL;
static Window windows[MAX_INSTANCES] = { 0 };

static GThread *x_event_thread = NULL;
static GMutex disp_lock;

static guint rrow, rcol, rcfg;
static gboolean rrowsel = FALSE, selecting = FALSE;

GST_DEBUG_CATEGORY (NVDS_APP);

GOptionEntry entries[] = {
  {"version", 'v', 0, G_OPTION_ARG_NONE, &print_version,
      "Print DeepStreamSDK version", NULL}
  ,
  {"tiledtext", 't', 0, G_OPTION_ARG_NONE, &show_bbox_text,
      "Display Bounding box labels in tiled mode", NULL}
  ,
  {"version-all", 0, 0, G_OPTION_ARG_NONE, &print_dependencies_version,
      "Print DeepStreamSDK and dependencies version", NULL}
  ,
  {"cfg-file", 'c', 0, G_OPTION_ARG_FILENAME_ARRAY, &cfg_files,
      "Set the config file", NULL}
  ,
  {"input-uri", 'i', 0, G_OPTION_ARG_FILENAME_ARRAY, &input_uris,
      "Set the input uri (file://stream or rtsp://stream)", NULL}
  ,
  {NULL}
  ,
};

typedef enum NvDsAnalyticType {
   NVDS_ANALYTIC_OBJ_IN_ROI,
   NVDS_ANALYTIC_LINE_CROSSING_CUMULATIVE,
   NVDS_ANALYTIC_LINE_CROSSING_CURRENT_FRAME,
   NVDS_ANALYTIC_OVERCROWDING_STATUS,
 } NvDsAnalyticType;

typedef struct NvDsAnalyticObject {
   NvDsAnalyticType type;      
   gchar *name;      
   gchar *value;     
 } NvDsAnalyticObject;

static void
generate_analytic_object_meta (gpointer data, NvDsAnalyticType analytic_type, std::string name, guint value)
{
  NvDsAnalyticObject *obj = (NvDsAnalyticObject *) data;

  obj->type = analytic_type;
  obj->name = NULL;
  obj->value = NULL;

  switch (analytic_type)
  {
  case NVDS_ANALYTIC_OBJ_IN_ROI:
  case NVDS_ANALYTIC_LINE_CROSSING_CUMULATIVE:
  case NVDS_ANALYTIC_LINE_CROSSING_CURRENT_FRAME:
    obj->name = g_strdup (name.c_str());
    obj->value = g_strdup_printf("%d", value);
    break;

  case NVDS_ANALYTIC_OVERCROWDING_STATUS:
    obj->name = g_strdup (name.c_str());
    if (value) {
      obj->value = g_strdup ("false");
    } else {
      obj->value = g_strdup ("true");
    }
    break;

  default:
    g_print ("invalid analytic type\n");
    break;
  }
}

static void
generate_product_object_meta (gpointer data, NvDsAnalyticType analytic_type, std::string name, guint value)
{
  NvDsProductObject *obj = (NvDsProductObject *) data;

  obj->type = NULL;
  obj->brand = NULL;
  obj->shape = NULL;

  switch (analytic_type)
  {
  case NVDS_ANALYTIC_OBJ_IN_ROI:
    obj->type = g_strdup ("object_in_roi");
    obj->brand = g_strdup (name.c_str());
    obj->shape = g_strdup_printf("%d", value);
    break;
  case NVDS_ANALYTIC_LINE_CROSSING_CUMULATIVE:
    obj->type = g_strdup ("line_crossing_cumulative");
    obj->brand = g_strdup (name.c_str());
    obj->shape = g_strdup_printf("%d", value);
    break;
  case NVDS_ANALYTIC_LINE_CROSSING_CURRENT_FRAME:
    obj->type = g_strdup ("line_crossing_current_frame");
    obj->brand = g_strdup (name.c_str());
    obj->shape = g_strdup_printf("%d", value);
    break;

  case NVDS_ANALYTIC_OVERCROWDING_STATUS:
    obj->type = g_strdup ("overcrowding_status");
    obj->brand = g_strdup (name.c_str());
    if (value) {
      obj->shape = g_strdup ("false");
    } else {
      obj->shape = g_strdup ("true");
    }
    break;

  default:
    g_print ("invalid analytic type\n");
    break;
  }
}

/**
 * Callback function to be called once all inferences (Primary + Secondary)
 * are done. This is opportunity to modify content of the metadata.
 * e.g. Here Person is being replaced with Man/Woman and corresponding counts
 * are being maintained. It should be modified according to network classes
 * or can be removed altogether if not required.
 */
static void
all_bbox_generated (AppCtx * appCtx, GstBuffer * buf,
    NvDsBatchMeta * batch_meta, guint index)
{
  guint num_male = 0;
  guint num_female = 0;
  guint num_objects[128];

  int count = 0;

  memset (num_objects, 0, sizeof (num_objects));

  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
    for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {
      NvDsObjectMeta *obj = (NvDsObjectMeta *) l_obj->data;
      if (obj->unique_component_id ==
          (gint) appCtx->config.primary_gie_config.unique_id) {
        if (obj->class_id >= 0 && obj->class_id < 128) {
          num_objects[obj->class_id]++;
        }
        if (appCtx->person_class_id > -1
            && obj->class_id == appCtx->person_class_id) {
          if (strstr (obj->text_params.display_text, "Man")) {
            str_replace (obj->text_params.display_text, "Man", "");
            str_replace (obj->text_params.display_text, "Person", "Man");
            num_male++;
          } else if (strstr (obj->text_params.display_text, "Woman")) {
            str_replace (obj->text_params.display_text, "Woman", "");
            str_replace (obj->text_params.display_text, "Person", "Woman");
            num_female++;
          }
        }
      }
      count++;
    }
  }
}

static void
generate_ts_rfc3339 (char *buf, int buf_size)
{
  time_t tloc;
  struct tm tm_log;
  struct timespec ts;
  char strmsec[6];              //.nnnZ\0

  clock_gettime (CLOCK_REALTIME, &ts);
  memcpy (&tloc, (void *) (&ts.tv_sec), sizeof (time_t));
  gmtime_r (&tloc, &tm_log);
  strftime (buf, buf_size, "%Y-%m-%dT%H:%M:%S", &tm_log);
  int ms = ts.tv_nsec / 1000000;
  g_snprintf (strmsec, sizeof (strmsec), ".%.3dZ", ms);
  strncat (buf, strmsec, buf_size);
}

static gpointer
meta_copy_func (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;
  NvDsEventMsgMeta *dstMeta = NULL;

  dstMeta = (NvDsEventMsgMeta *) g_memdup2 (srcMeta, sizeof (NvDsEventMsgMeta));

  if (srcMeta->ts)
    dstMeta->ts = g_strdup (srcMeta->ts);

  if (srcMeta->objSignature.size > 0) {
    dstMeta->objSignature.signature = (gdouble *) g_memdup2 (srcMeta->objSignature.signature,
        srcMeta->objSignature.size);
    dstMeta->objSignature.size = srcMeta->objSignature.size;
  }

  if (srcMeta->objectId) {
    dstMeta->objectId = g_strdup (srcMeta->objectId);
  }

  if (srcMeta->sensorStr) {
    dstMeta->sensorStr = g_strdup (srcMeta->sensorStr);
  }

  if (srcMeta->extMsgSize > 0) {
    if (srcMeta->objType == NVDS_OBJECT_TYPE_VEHICLE) {
      NvDsVehicleObject *srcObj = (NvDsVehicleObject *) srcMeta->extMsg;
      NvDsVehicleObject *obj =
          (NvDsVehicleObject *) g_malloc0 (sizeof (NvDsVehicleObject));
      if (srcObj->type)
        obj->type = g_strdup (srcObj->type);
      if (srcObj->make)
        obj->make = g_strdup (srcObj->make);
      if (srcObj->model)
        obj->model = g_strdup (srcObj->model);
      if (srcObj->color)
        obj->color = g_strdup (srcObj->color);
      if (srcObj->license)
        obj->license = g_strdup (srcObj->license);
      if (srcObj->region)
        obj->region = g_strdup (srcObj->region);

      dstMeta->extMsg = obj;
      dstMeta->extMsgSize = sizeof (NvDsVehicleObject);
    } else if (srcMeta->objType == NVDS_OBJECT_TYPE_PERSON) {
      NvDsPersonObject *srcObj = (NvDsPersonObject *) srcMeta->extMsg;
      NvDsPersonObject *obj =
          (NvDsPersonObject *) g_malloc0 (sizeof (NvDsPersonObject));

      obj->age = srcObj->age;

      if (srcObj->gender)
        obj->gender = g_strdup (srcObj->gender);
      if (srcObj->cap)
        obj->cap = g_strdup (srcObj->cap);
      if (srcObj->hair)
        obj->hair = g_strdup (srcObj->hair);
      if (srcObj->apparel)
        obj->apparel = g_strdup (srcObj->apparel);

      dstMeta->extMsg = obj;
      dstMeta->extMsgSize = sizeof (NvDsPersonObject);
    } else if (srcMeta->objType == NVDS_OBJECT_TYPE_CUSTOM) {
      NvDsAnalyticObject *srcObj = (NvDsAnalyticObject *) srcMeta->extMsg;
      NvDsAnalyticObject *obj =
          (NvDsAnalyticObject *) g_malloc0 (sizeof (NvDsAnalyticObject));

      obj->type = srcObj->type;

      if (srcObj->name)
        obj->name = g_strdup (srcObj->name);
      if (srcObj->value)
        obj->value = g_strdup (srcObj->value);

      dstMeta->extMsg = obj;
      dstMeta->extMsgSize = sizeof (NvDsAnalyticObject);
    } else if (srcMeta->objType == NVDS_OBJECT_TYPE_PRODUCT) {
      NvDsProductObject *srcObj = (NvDsProductObject *) srcMeta->extMsg;
      NvDsProductObject *obj =
          (NvDsProductObject *) g_malloc0 (sizeof (NvDsProductObject));

      if (srcObj->type)
        obj->type = g_strdup (srcObj->type);
      if (srcObj->brand)
        obj->brand = g_strdup (srcObj->brand);
      if (srcObj->shape)
        obj->shape = g_strdup (srcObj->shape);

      dstMeta->extMsg = obj;
      dstMeta->extMsgSize = sizeof (NvDsProductObject);
    }
  }

  return dstMeta;
}

static void
meta_free_func (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;
  user_meta->user_meta_data = NULL;

  if (srcMeta->ts) {
    g_free (srcMeta->ts);
  }

  if (srcMeta->objSignature.size > 0) {
    g_free (srcMeta->objSignature.signature);
    srcMeta->objSignature.size = 0;
  }

  if (srcMeta->objectId) {
    g_free (srcMeta->objectId);
  }

  if (srcMeta->sensorStr) {
    g_free (srcMeta->sensorStr);
  }

  if (srcMeta->extMsgSize > 0) {
    if (srcMeta->objType == NVDS_OBJECT_TYPE_VEHICLE) {
      NvDsVehicleObject *obj = (NvDsVehicleObject *) srcMeta->extMsg;
      if (obj->type)
        g_free (obj->type);
      if (obj->color)
        g_free (obj->color);
      if (obj->make)
        g_free (obj->make);
      if (obj->model)
        g_free (obj->model);
      if (obj->license)
        g_free (obj->license);
      if (obj->region)
        g_free (obj->region);
    } else if (srcMeta->objType == NVDS_OBJECT_TYPE_PERSON) {
      NvDsPersonObject *obj = (NvDsPersonObject *) srcMeta->extMsg;

      if (obj->gender)
        g_free (obj->gender);
      if (obj->cap)
        g_free (obj->cap);
      if (obj->hair)
        g_free (obj->hair);
      if (obj->apparel)
        g_free (obj->apparel);
    } else if (srcMeta->objType == NVDS_OBJECT_TYPE_CUSTOM) {
      NvDsAnalyticObject *obj = (NvDsAnalyticObject *) srcMeta->extMsg;

      if (obj->name)
        g_free (obj->name);
      if (obj->value)
        g_free (obj->value);
    } else if (srcMeta->objType == NVDS_OBJECT_TYPE_PRODUCT) {
      NvDsProductObject *obj = (NvDsProductObject *) srcMeta->extMsg;

      if (obj->type)
        g_free (obj->type);
      if (obj->brand)
        g_free (obj->brand);
      if (obj->shape)
        g_free (obj->shape);
    }
    g_free (srcMeta->extMsg);
    srcMeta->extMsg = NULL;
    srcMeta->extMsgSize = 0;
  }
  g_free (srcMeta);
}

static gpointer
meta_copy_func_custom (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsCustomMsgInfo *srcMeta = (NvDsCustomMsgInfo *) user_meta->user_meta_data;
  NvDsCustomMsgInfo *dstMeta = NULL;

  dstMeta = (NvDsCustomMsgInfo *) g_memdup2 (srcMeta, sizeof (NvDsCustomMsgInfo));

  if (srcMeta->message)
    dstMeta->message = (gpointer) g_strdup ((const char*)srcMeta->message);
  dstMeta->size = srcMeta->size;

  return dstMeta;
}

static void
meta_free_func_custom (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsCustomMsgInfo *srcMeta = (NvDsCustomMsgInfo *) user_meta->user_meta_data;

  if (srcMeta->message)
    g_free (srcMeta->message);
  srcMeta->size = 0;

  g_free (user_meta->user_meta_data);
}

static void
generate_event_msg_meta (AppCtx * appCtx, gpointer data, gboolean useTs,
    GstClockTime ts, gchar * src_uri, gint stream_id, guint sensor_id,
    NvDsProductObject * analytic_obj, NvDsFrameMeta * frame_meta)
{
  NvDsEventMsgMeta *meta = (NvDsEventMsgMeta *) data;

  //meta->type = NVDS_EVENT_CUSTOM;
  meta->type = NVDS_EVENT_MOVING;
  //meta->objType = NVDS_OBJECT_TYPE_CUSTOM; 
  meta->objType = NVDS_OBJECT_TYPE_PRODUCT;
  meta->sensorId = sensor_id;
  meta->placeId = sensor_id;
  meta->moduleId = sensor_id;
  meta->frameId = frame_meta->frame_num;
  meta->ts = (gchar *) g_malloc0 (MAX_TIME_STAMP_LEN + 1);

  /** INFO: This API is called once for every 30 frames (now) */
  if ((useTs && src_uri) || appCtx->config.source_attr_all_config.type == NV_DS_SOURCE_IPC) {
    g_snprintf (meta->ts, sizeof (meta->ts), ".%ld", ts);
  } else {
    generate_ts_rfc3339 (meta->ts, MAX_TIME_STAMP_LEN);
  }

  meta->extMsg = analytic_obj;
  meta->extMsgSize = sizeof (NvDsProductObject);
}

static void
add_event_msg_meta (AppCtx * appCtx, gboolean useTs, GstClockTime ts,
    gint stream_id, NvDsProductObject * analytic_obj, NvDsFrameMeta * frame_meta, NvDsBatchMeta * batch_meta)
{
  NvDsEventMsgMeta *msg_meta =
      (NvDsEventMsgMeta *)g_malloc0(sizeof(NvDsEventMsgMeta));
  generate_event_msg_meta(appCtx, msg_meta, useTs, ts,
                          appCtx->config.multi_source_config[stream_id].uri, stream_id,
                          appCtx->config.multi_source_config[stream_id].camera_id,
                          analytic_obj, frame_meta);

  NvDsUserMeta *user_event_meta =
      nvds_acquire_user_meta_from_pool(batch_meta);
  if (!user_event_meta) {
    g_print("Error in attaching event meta to buffer\n");
    return;
  }

  user_event_meta->user_meta_data = (void *)msg_meta;
  user_event_meta->base_meta.batch_meta = batch_meta;
  user_event_meta->base_meta.meta_type = NVDS_EVENT_MSG_META;
  user_event_meta->base_meta.copy_func =
      (NvDsMetaCopyFunc)meta_copy_func;
  user_event_meta->base_meta.release_func =
      (NvDsMetaReleaseFunc)meta_free_func;
  nvds_add_user_meta_to_frame(frame_meta, user_event_meta);
}

static void
add_custom_msg_meta(AppCtx *appCtx, gint stream_id,
    NvDsProductObject *obj, NvDsFrameMeta *frame_meta, NvDsBatchMeta *batch_meta)
{
  NvDsCustomMsgInfo *msg_meta =
      (NvDsCustomMsgInfo *)g_malloc0(sizeof(NvDsCustomMsgInfo));

  gchar *message_data;
  message_data = g_strdup_printf("%d|%s|%s|%s", stream_id, obj->type, obj->brand, obj->shape);
  msg_meta->size = strlen(message_data);
  msg_meta->message = g_strdup(message_data);

  NvDsUserMeta *user_event_meta =
      nvds_acquire_user_meta_from_pool(batch_meta);
  if (user_event_meta) {
    user_event_meta->user_meta_data = (void *)msg_meta;
    user_event_meta->base_meta.batch_meta = batch_meta;
    user_event_meta->base_meta.meta_type = NVDS_CUSTOM_MSG_BLOB;
    user_event_meta->base_meta.copy_func =
        (NvDsMetaCopyFunc)meta_copy_func_custom;
    user_event_meta->base_meta.release_func =
        (NvDsMetaReleaseFunc)meta_free_func_custom;
    nvds_add_user_meta_to_frame(frame_meta, user_event_meta);
  } else {
    g_print("Error in attaching event meta to buffer\n");
  }

  if (obj->type)
    g_free(obj->type);
  if (obj->brand)
    g_free(obj->brand);
  if (obj->shape)
    g_free(obj->shape);
  g_free(obj);
}

/**
 * Callback function to be called once all inferences (Primary + Secondary)
 * are done. This is opportunity to modify content of the metadata.
 * e.g. Here Person is being replaced with Man/Woman and corresponding counts
 * are being maintained. It should be modified according to network classes
 * or can be removed altogether if not required.
 */
static void
bbox_generated_probe_after_analytics (AppCtx * appCtx, GstBuffer * buf,
    NvDsBatchMeta * batch_meta, guint index)
{
  NvDsUserMeta *usr_meta = NULL;
  NvDsAnalyticsFrameMeta *analytic_meta = NULL;
  GstClockTime buffer_pts = 0;
  guint32 stream_id = 0;

  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
    stream_id = frame_meta->source_id;

    GList *l;
    for (l = frame_meta->frame_user_meta_list; l != NULL; l = l->next) {
      usr_meta = (NvDsUserMeta *) (l->data);

      if (usr_meta->base_meta.meta_type != NVDS_USER_FRAME_META_NVDSANALYTICS)
        continue;

      analytic_meta = (NvDsAnalyticsFrameMeta *) usr_meta->user_meta_data;

      for (std::pair<std::string, uint32_t> status : analytic_meta->objInROIcnt) {
        LOGD("key : %s , value : %d\n", status.first.c_str(), status.second);
        NvDsProductObject *obj =
            (NvDsProductObject *) g_malloc0 (sizeof (NvDsProductObject));
        generate_product_object_meta(obj, NVDS_ANALYTIC_OBJ_IN_ROI, status.first, status.second);

        add_custom_msg_meta(appCtx, stream_id, obj, frame_meta, batch_meta);
        //add_event_msg_meta(appCtx, False, buffer_pts, stream_id, obj, frame_meta, batch_meta);
      } 

      for (std::pair<std::string, uint32_t> status : analytic_meta->objLCCumCnt) {
        LOGD("key : %s , value : %d\n", status.first.c_str(), status.second);
        NvDsProductObject *obj =
            (NvDsProductObject *) g_malloc0 (sizeof (NvDsProductObject));
        generate_product_object_meta(obj, NVDS_ANALYTIC_LINE_CROSSING_CUMULATIVE, status.first, status.second - count_offset[stream_id]);

        add_custom_msg_meta(appCtx, stream_id, obj, frame_meta, batch_meta);

        // Memory current cumulative
        count_current[stream_id] = status.second;
        //add_event_msg_meta(appCtx, False, buffer_pts, stream_id, obj, frame_meta, batch_meta);
      }
      for (std::pair<std::string, uint32_t> status : analytic_meta->objLCCurrCnt) {
        LOGD("key : %s , value : %d\n", status.first.c_str(), status.second);
        NvDsProductObject *obj =
            (NvDsProductObject *) g_malloc0 (sizeof (NvDsProductObject));
        generate_product_object_meta(obj, NVDS_ANALYTIC_LINE_CROSSING_CURRENT_FRAME, status.first, status.second);

        add_custom_msg_meta(appCtx, stream_id, obj, frame_meta, batch_meta);
        //add_event_msg_meta(appCtx, False, buffer_pts, stream_id, obj, frame_meta, batch_meta);
      } 
      for (std::pair<std::string, bool> status : analytic_meta->ocStatus) {
        LOGD("key : %s , value : %d\n", status.first.c_str(), status.second);
        NvDsProductObject *obj =
            (NvDsProductObject *) g_malloc0 (sizeof (NvDsProductObject));
        generate_product_object_meta(obj, NVDS_ANALYTIC_OVERCROWDING_STATUS, status.first, status.second);

        add_custom_msg_meta(appCtx, stream_id, obj, frame_meta, batch_meta);
        //add_event_msg_meta(appCtx, False, buffer_pts, stream_id, obj, frame_meta, batch_meta);
      }
    }
    //testAppCtx->streams[stream_id].frameCount++;
  }
}

/**
 * Function to handle program interrupt signal.
 * It installs default handler after handling the interrupt.
 */
static void
_intr_handler (int signum)
{
  struct sigaction action;

  NVGSTDS_ERR_MSG_V ("User Interrupted.. \n");

  memset (&action, 0, sizeof (action));
  action.sa_handler = SIG_DFL;

  sigaction (SIGINT, &action, NULL);

  cintr = TRUE;
}

/**
 * callback function to print the performance numbers of each stream.
 */
static void
perf_cb (gpointer context, NvDsAppPerfStruct * str)
{
  static guint header_print_cnt = 0;
  guint i;
  AppCtx *appCtx = (AppCtx *) context;
  guint numf = str->num_instances;

  g_mutex_lock (&fps_lock);
  for (i = 0; i < numf; i++) {
    fps[i] = str->fps[i];
    fps_avg[i] = str->fps_avg[i];
  }

  if (header_print_cnt % 20 == 0) {
    g_print ("\n**PERF:  ");
    for (i = 0; i < numf; i++) {
      g_print ("FPS %d (Avg)\t", i);
    }
    g_print ("\n");
    header_print_cnt = 0;
  }
  header_print_cnt++;
  if (num_instances > 1)
    g_print ("PERF(%d): ", appCtx->index);
  else
    g_print ("**PERF:  ");

  for (i = 0; i < numf; i++) {
    g_print ("%.2f (%.2f)\t", fps[i], fps_avg[i]);
  }
  g_print ("\n");
  g_mutex_unlock (&fps_lock);
}

/**
 * Loop function to check the status of interrupts.
 * It comes out of loop if application got interrupted.
 */
static gboolean
check_for_interrupt (gpointer data)
{
  if (quit) {
    return FALSE;
  }

  if (cintr) {
    cintr = FALSE;

    quit = TRUE;
    g_main_loop_quit (main_loop);

    return FALSE;
  }
  return TRUE;
}

/*
 * Function to install custom handler for program interrupt signal.
 */
static void
_intr_setup (void)
{
  struct sigaction action;

  memset (&action, 0, sizeof (action));
  action.sa_handler = _intr_handler;

  sigaction (SIGINT, &action, NULL);
}

static gboolean
kbhit (void)
{
  struct timeval tv;
  fd_set rdfs;

  tv.tv_sec = 0;
  tv.tv_usec = 0;

  FD_ZERO (&rdfs);
  FD_SET (STDIN_FILENO, &rdfs);

  select (STDIN_FILENO + 1, &rdfs, NULL, NULL, &tv);
  return FD_ISSET (STDIN_FILENO, &rdfs);
}

/*
 * Function to enable / disable the canonical mode of terminal.
 * In non canonical mode input is available immediately (without the user
 * having to type a line-delimiter character).
 */
static void
changemode (int dir)
{
  static struct termios oldt, newt;

  if (dir == 1) {
    tcgetattr (STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON);
    tcsetattr (STDIN_FILENO, TCSANOW, &newt);
  } else
    tcsetattr (STDIN_FILENO, TCSANOW, &oldt);
}

static void
print_runtime_commands (void)
{
  g_print ("\nRuntime commands:\n"
      "\th: Print this help\n"
      "\tq: Quit\n\n" "\tp: Pause\n" "\tr: Resume\n\n");

  if (appCtx[0]->config.tiled_display_config.enable) {
    g_print
        ("NOTE: To expand a source in the 2D tiled display and view object details,"
        " left-click on the source.\n"
        "      To go back to the tiled display, right-click anywhere on the window.\n\n");
  }
}

/**
 * Loop function to check keyboard inputs and status of each pipeline.
 */
static gboolean
event_thread_func (gpointer arg)
{
  guint i;
  gboolean ret = TRUE;

  // Check if all instances have quit
  for (i = 0; i < num_instances; i++) {
    if (!appCtx[i]->quit)
      break;
  }

  if (i == num_instances) {
    quit = TRUE;
    g_main_loop_quit (main_loop);
    return FALSE;
  }
  // Check for keyboard input
  if (!kbhit ()) {
    //continue;
    return TRUE;
  }
  int c = fgetc (stdin);
  g_print ("\n");

  gint source_id;
  GstElement *tiler = appCtx[rcfg]->pipeline.tiled_display_bin.tiler;
  if (appCtx[rcfg]->config.tiled_display_config.enable)
  {
    g_object_get (G_OBJECT (tiler), "show-source", &source_id, NULL);

    if (selecting) {
      if (rrowsel == FALSE) {
        if (c >= '0' && c <= '9') {
          rrow = c - '0';
          if (rrow < appCtx[rcfg]->config.tiled_display_config.rows){
            g_print ("--selecting source  row %d--\n", rrow);
            rrowsel = TRUE;
          }else{
            g_print ("--selected source  row %d out of bound, reenter\n", rrow);
          }
        }
      } else {
        if (c >= '0' && c <= '9') {
          unsigned int tile_num_columns = appCtx[rcfg]->config.tiled_display_config.columns;
          rcol = c - '0';
          if (rcol < tile_num_columns){
            selecting = FALSE;
            rrowsel = FALSE;
            source_id = tile_num_columns * rrow + rcol;
            g_print ("--selecting source  col %d sou=%d--\n", rcol, source_id);
            if (source_id >= (gint) appCtx[rcfg]->config.num_source_sub_bins) {
              source_id = -1;
            } else {
              appCtx[rcfg]->show_bbox_text = TRUE;
              appCtx[rcfg]->active_source_index = source_id;
              g_object_set (G_OBJECT (tiler), "show-source", source_id, NULL);
            }
          }else{
            g_print ("--selected source  col %d out of bound, reenter\n", rcol);
          }
        }
      }
    }
  }
  switch (c) {
    case 'h':
      print_runtime_commands ();
      break;
    case 'p':
      for (i = 0; i < num_instances; i++)
        pause_pipeline (appCtx[i]);
      break;
    case 'r':
      for (i = 0; i < num_instances; i++)
        resume_pipeline (appCtx[i]);
      break;
    case 'q':
      quit = TRUE;
      g_main_loop_quit (main_loop);
      ret = FALSE;
      break;
    case 'c':
      if (appCtx[rcfg]->config.tiled_display_config.enable && selecting == FALSE && source_id == -1) {
        g_print ("--selecting config file --\n");
        c = fgetc (stdin);
        if (c >= '0' && c <= '9') {
          rcfg = c - '0';
          if (rcfg < num_instances) {
            g_print ("--selecting config  %d--\n", rcfg);
          } else {
            g_print ("--selected config file %d out of bound, reenter\n", rcfg);
            rcfg = 0;
          }
        }
      }
      break;
    case 'z':
      if (appCtx[rcfg]->config.tiled_display_config.enable && source_id == -1 && selecting == FALSE) {
        g_print ("--selecting source --\n");
        selecting = TRUE;
      } else {
        if (!show_bbox_text)
          appCtx[rcfg]->show_bbox_text = FALSE;
        g_object_set (G_OBJECT (tiler), "show-source", -1, NULL);
        appCtx[rcfg]->active_source_index = -1;
        selecting = FALSE;
        rcfg = 0;
        g_print ("--tiled mode --\n");
      }
      break;
    default:
      break;
  }
  return ret;
}

static int
get_source_id_from_coordinates (float x_rel, float y_rel, AppCtx *appCtx)
{
  int tile_num_rows = appCtx->config.tiled_display_config.rows;
  int tile_num_columns = appCtx->config.tiled_display_config.columns;

  int source_id = (int) (x_rel * tile_num_columns);
  source_id += ((int) (y_rel * tile_num_rows)) * tile_num_columns;

  /* Don't allow clicks on empty tiles. */
  if (source_id >= (gint) appCtx->config.num_source_sub_bins)
    source_id = -1;

  return source_id;
}

/**
 * Thread to monitor X window events.
 */
static gpointer
nvds_x_event_thread (gpointer data)
{
  g_mutex_lock (&disp_lock);
  while (display) {
    XEvent e;
    guint index;
    memset(&e, 0, sizeof(XEvent));
    while (XPending (display)) {
      XNextEvent (display, &e);
      switch (e.type) {
        case ButtonPress:
        {
          XWindowAttributes win_attr;
          XButtonEvent ev = e.xbutton;
          gint source_id;
          GstElement *tiler;
          memset(&win_attr, 0, sizeof(XWindowAttributes));

          XGetWindowAttributes (display, ev.window, &win_attr);

          for (index = 0; index < MAX_INSTANCES; index++)
            if (ev.window == windows[index])
              break;

          tiler = appCtx[index]->pipeline.tiled_display_bin.tiler;
          g_object_get (G_OBJECT (tiler), "show-source", &source_id, NULL);

          if (ev.button == Button1 && source_id == -1  && (index >=0 && index < MAX_INSTANCES )) {
            source_id =
                get_source_id_from_coordinates (ev.x * 1.0 / win_attr.width,
                ev.y * 1.0 / win_attr.height, appCtx[index]);
            if (source_id > -1) {
              g_object_set (G_OBJECT (tiler), "show-source", source_id, NULL);
              appCtx[index]->active_source_index = source_id;
              appCtx[index]->show_bbox_text = TRUE;
            }
          } else if (ev.button == Button3) {
            g_object_set (G_OBJECT (tiler), "show-source", -1, NULL);
            appCtx[index]->active_source_index = -1;
            if (!show_bbox_text)
              appCtx[index]->show_bbox_text = FALSE;
          }
        }
          break;
        case KeyRelease:
        case KeyPress:
        {
          KeySym p, r, q;
          guint i;
          p = XKeysymToKeycode (display, XK_P);
          r = XKeysymToKeycode (display, XK_R);
          q = XKeysymToKeycode (display, XK_Q);
          if (e.xkey.keycode == p) {
            for (i = 0; i < num_instances; i++)
              pause_pipeline (appCtx[i]);
            break;
          }
          if (e.xkey.keycode == r) {
            for (i = 0; i < num_instances; i++)
              resume_pipeline (appCtx[i]);
            break;
          }
          if (e.xkey.keycode == q) {
            quit = TRUE;
            g_main_loop_quit (main_loop);
          }
        }
          break;
        case ClientMessage:
        {
          Atom wm_delete;
          for (index = 0; index < MAX_INSTANCES; index++)
            if (e.xclient.window == windows[index])
              break;

          wm_delete = XInternAtom (display, "WM_DELETE_WINDOW", 1);
          if (wm_delete != None && wm_delete == (Atom) e.xclient.data.l[0]) {
            quit = TRUE;
            g_main_loop_quit (main_loop);
          }
        }
          break;
      }
    }
    g_mutex_unlock (&disp_lock);
    g_usleep (G_USEC_PER_SEC / 20);
    g_mutex_lock (&disp_lock);
  }
  g_mutex_unlock (&disp_lock);
  return NULL;
}

/**
 * callback function to add application specific metadata.
 * Here it demonstrates how to display the URI of source in addition to
 * the text generated after inference.
 */
static gboolean
overlay_graphics (AppCtx * appCtx, GstBuffer * buf,
    NvDsBatchMeta * batch_meta, guint index)
{
  int srcIndex = appCtx->active_source_index;
  if (srcIndex == -1)
    return TRUE;

  NvDsFrameLatencyInfo *latency_info = NULL;
  NvDsDisplayMeta *display_meta =
      nvds_acquire_display_meta_from_pool (batch_meta);

  display_meta->num_labels = 1;
  display_meta->text_params[0].display_text = g_strdup_printf ("Source: %s",
      appCtx->config.multi_source_config[srcIndex].uri);

  display_meta->text_params[0].y_offset = 20;
  display_meta->text_params[0].x_offset = 20;
  display_meta->text_params[0].font_params.font_color = (NvOSD_ColorParams) {
  0, 1, 0, 1};
  display_meta->text_params[0].font_params.font_size =
      appCtx->config.osd_config.text_size * 1.5;
  display_meta->text_params[0].font_params.font_name = "Serif";
  display_meta->text_params[0].set_bg_clr = 1;
  display_meta->text_params[0].text_bg_clr = (NvOSD_ColorParams) {
  0, 0, 0, 1.0};


  if(nvds_enable_latency_measurement) {
    g_mutex_lock (&appCtx->latency_lock);
    latency_info = &appCtx->latency_info[index];
    display_meta->num_labels++;
    display_meta->text_params[1].display_text = g_strdup_printf ("Latency: %lf",
        latency_info->latency);
    g_mutex_unlock (&appCtx->latency_lock);

    display_meta->text_params[1].y_offset = (display_meta->text_params[0].y_offset * 2 )+
      display_meta->text_params[0].font_params.font_size;
    display_meta->text_params[1].x_offset = 20;
    display_meta->text_params[1].font_params.font_color = (NvOSD_ColorParams) {
      0, 1, 0, 1};
    display_meta->text_params[1].font_params.font_size =
      appCtx->config.osd_config.text_size * 1.5;
    display_meta->text_params[1].font_params.font_name = "Arial";
    display_meta->text_params[1].set_bg_clr = 1;
    display_meta->text_params[1].text_bg_clr = (NvOSD_ColorParams) {
      0, 0, 0, 1.0};
  }

  //nvds_add_display_meta_to_frame (nvds_get_nth_frame_meta (batch_meta->frame_meta_list, 0), display_meta);
  return TRUE;
}

static void
subscribe_cb (NvMsgBrokerErrorType flag, void *msg, int msg_len, char *topic,
    void *uData)
{
  JsonNode *rootNode = NULL;
  GError *error = NULL;
  gboolean ret;

  if (flag == NV_MSGBROKER_API_ERR) {
    NVGSTDS_ERR_MSG_V ("Error in consuming message.");
  } else {
    GST_DEBUG ("Consuming message, on topic[%s]. Payload =%.*s\n\n", topic,
        msg_len, (char *) msg);
  }

  JsonParser *parser = json_parser_new ();
  ret = json_parser_load_from_data (parser, (char *) msg, msg_len, &error);
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("Error in parsing json message %s", error->message);
    g_error_free (error);
    g_object_unref (parser);
    return;
  }

   /**
   * Following minimum json message is expected to trigger the start / stop
   * of smart record.
   * {
   *   command: string   // "reset"
   *   stream_id: int     // 0
   * }
   */

  rootNode = json_parser_get_root (parser);
  if (JSON_NODE_HOLDS_OBJECT (rootNode)) {
    JsonObject *object;

    object = json_node_get_object (rootNode);
    if (json_object_has_member (object, "command") && json_object_has_member (object, "stream_id")) {
      const gchar *type = json_object_get_string_member (object, "command");
      const gint64 stream_id = json_object_get_int_member (object, "stream_id");
      if (!g_strcmp0 (type, "reset")) {
          count_offset[stream_id] = count_current[stream_id];
      }
    }
  }

error:
  g_object_unref (parser);
  return;
}

static gboolean
recreate_pipeline_thread_func (gpointer arg)
{
  guint i;
  gboolean ret = TRUE;
  AppCtx *appCtx = (AppCtx *) arg;

  g_print ("Destroy pipeline\n");
  destroy_pipeline (appCtx);

  g_print ("Recreate pipeline\n");
  if (!create_pipeline_with_subscribe (appCtx, bbox_generated_probe_after_analytics,
          all_bbox_generated, perf_cb, overlay_graphics, subscribe_cb)) {
    NVGSTDS_ERR_MSG_V ("Failed to create pipeline");
    return_value = -1;
    return FALSE;
  }

  if (gst_element_set_state (appCtx->pipeline.pipeline,
          GST_STATE_PAUSED) == GST_STATE_CHANGE_FAILURE) {
    NVGSTDS_ERR_MSG_V ("Failed to set pipeline to PAUSED");
    return_value = -1;
    return FALSE;
  }

  for (i = 0; i < appCtx->config.num_sink_sub_bins; i++) {
    if (!GST_IS_VIDEO_OVERLAY (appCtx->pipeline.instance_bins[0].sink_bin.
            sub_bins[i].sink)) {
      continue;
    }

    gst_video_overlay_set_window_handle (GST_VIDEO_OVERLAY (appCtx->pipeline.
            instance_bins[0].sink_bin.sub_bins[i].sink),
        (gulong) windows[appCtx->index]);
    gst_video_overlay_expose (GST_VIDEO_OVERLAY (appCtx->pipeline.
            instance_bins[0].sink_bin.sub_bins[i].sink));
  }

  if (gst_element_set_state (appCtx->pipeline.pipeline,
          GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {

    g_print ("\ncan't set pipeline to playing state.\n");
    return_value = -1;
    return FALSE;
  }

  return ret;
}

int
main (int argc, char *argv[])
{
  GOptionContext *ctx = NULL;
  GOptionGroup *group = NULL;
  GError *error = NULL;
  guint i;

  ctx = g_option_context_new ("Nvidia DeepStream Demo");
  group = g_option_group_new ("abc", NULL, NULL, NULL, NULL);
  g_option_group_add_entries (group, entries);

  g_option_context_set_main_group (ctx, group);
  g_option_context_add_group (ctx, gst_init_get_option_group ());

  GST_DEBUG_CATEGORY_INIT (NVDS_APP, "NVDS_APP", 0, NULL);

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  if (!g_option_context_parse (ctx, &argc, &argv, &error)) {
    NVGSTDS_ERR_MSG_V ("%s", error->message);
    return -1;
  }

  if (print_version) {
    g_print ("deepstream-app version %d.%d.%d\n",
        NVDS_APP_VERSION_MAJOR, NVDS_APP_VERSION_MINOR, NVDS_APP_VERSION_MICRO);
    nvds_version_print ();
    return 0;
  }

  if (print_dependencies_version) {
    g_print ("deepstream-app version %d.%d.%d\n",
        NVDS_APP_VERSION_MAJOR, NVDS_APP_VERSION_MINOR, NVDS_APP_VERSION_MICRO);
    nvds_version_print ();
    nvds_dependencies_version_print ();
    return 0;
  }

  if (cfg_files) {
    num_instances = g_strv_length (cfg_files);
  }
  if (input_uris) {
    num_input_uris = g_strv_length (input_uris);
  }

  if (!cfg_files || num_instances == 0) {
    NVGSTDS_ERR_MSG_V ("Specify config file with -c option");
    return_value = -1;
    goto done;
  }

  for (i = 0; i < num_instances; i++) {
    appCtx[i] = (AppCtx *)g_malloc0 (sizeof (AppCtx));
    appCtx[i]->person_class_id = -1;
    appCtx[i]->car_class_id = -1;
    appCtx[i]->index = i;
    appCtx[i]->active_source_index = -1;
    if (show_bbox_text) {
      appCtx[i]->show_bbox_text = TRUE;
    }

    if (input_uris && input_uris[i]) {
      appCtx[i]->config.multi_source_config[0].uri =
          g_strdup_printf ("%s", input_uris[i]);
      g_free (input_uris[i]);
    }

    if(g_str_has_suffix(cfg_files[i], ".yml") ||
            g_str_has_suffix(cfg_files[i], ".yaml")) {
      if (!parse_config_file_yaml (&appCtx[i]->config, cfg_files[i])) {
        NVGSTDS_ERR_MSG_V ("Failed to parse config file '%s'", cfg_files[i]);
        appCtx[i]->return_value = -1;
        goto done;
      }
    } else if (g_str_has_suffix(cfg_files[i], ".txt")) {
      if (!parse_config_file (&appCtx[i]->config, cfg_files[i])) {
        NVGSTDS_ERR_MSG_V ("Failed to parse config file '%s'", cfg_files[i]);
        appCtx[i]->return_value = -1;
        goto done;
      }
    }
  }

  for (i = 0; i < num_instances; i++) {
    if (!create_pipeline_with_subscribe (appCtx[i], bbox_generated_probe_after_analytics,
            all_bbox_generated, perf_cb, overlay_graphics, subscribe_cb)) {
      NVGSTDS_ERR_MSG_V ("Failed to create pipeline");
      return_value = -1;
      goto done;
    }
  }

  main_loop = g_main_loop_new (NULL, FALSE);

  _intr_setup ();
  g_timeout_add (400, check_for_interrupt, NULL);

  g_mutex_init (&disp_lock);
  display = XOpenDisplay (NULL);
  for (i = 0; i < num_instances; i++) {
    guint j;
#if defined(__aarch64__)
      if (gst_element_set_state (appCtx[i]->pipeline.pipeline,
              GST_STATE_PAUSED) == GST_STATE_CHANGE_FAILURE) {
        NVGSTDS_ERR_MSG_V ("Failed to set pipeline to PAUSED");
        return_value = -1;
        goto done;
      }
#endif
    for (j = 0; j < appCtx[i]->config.num_sink_sub_bins; j++) {
      XTextProperty xproperty;
      gchar *title;
      guint width, height;
      XSizeHints hints = {0};

      if (!GST_IS_VIDEO_OVERLAY (appCtx[i]->pipeline.instance_bins[0].
              sink_bin.sub_bins[j].sink)) {
        continue;
      }

      if (!display) {
        NVGSTDS_ERR_MSG_V ("Could not open X Display");
        return_value = -1;
        goto done;
      }

      if (appCtx[i]->config.sink_bin_sub_bin_config[j].render_config.width)
        width =
            appCtx[i]->config.sink_bin_sub_bin_config[j].render_config.width;
      else
        width = appCtx[i]->config.tiled_display_config.width;

      if (appCtx[i]->config.sink_bin_sub_bin_config[j].render_config.height)
        height =
            appCtx[i]->config.sink_bin_sub_bin_config[j].render_config.height;
      else
        height = appCtx[i]->config.tiled_display_config.height;

      width = (width) ? width : DEFAULT_X_WINDOW_WIDTH;
      height = (height) ? height : DEFAULT_X_WINDOW_HEIGHT;

      hints.flags = PPosition | PSize;
      hints.x = appCtx[i]->config.sink_bin_sub_bin_config[j].render_config.offset_x;
      hints.y = appCtx[i]->config.sink_bin_sub_bin_config[j].render_config.offset_y;
      hints.width = width;
      hints.height = height;

      windows[i] =
          XCreateSimpleWindow (display, RootWindow (display,
              DefaultScreen (display)), hints.x, hints.y, width, height, 2,
              0x00000000, 0x00000000);

      XSetNormalHints(display, windows[i], &hints);

      if (num_instances > 1)
        title = g_strdup_printf (APP_TITLE "-%d", i);
      else
        title = g_strdup (APP_TITLE);
      if (XStringListToTextProperty ((char **) &title, 1, &xproperty) != 0) {
        XSetWMName (display, windows[i], &xproperty);
        XFree (xproperty.value);
      }

      XSetWindowAttributes attr = { 0 };
      if ((appCtx[i]->config.tiled_display_config.enable &&
              appCtx[i]->config.tiled_display_config.rows *
              appCtx[i]->config.tiled_display_config.columns == 1) ||
          (appCtx[i]->config.tiled_display_config.enable == 0)) {
        attr.event_mask = KeyPress;
      } else if (appCtx[i]->config.tiled_display_config.enable) {
        attr.event_mask = ButtonPress | KeyRelease;
      }
      XChangeWindowAttributes (display, windows[i], CWEventMask, &attr);

      Atom wmDeleteMessage = XInternAtom (display, "WM_DELETE_WINDOW", False);
      if (wmDeleteMessage != None) {
        XSetWMProtocols (display, windows[i], &wmDeleteMessage, 1);
      }
      XMapRaised (display, windows[i]);
      XSync (display, 1);       //discard the events for now
      gst_video_overlay_set_window_handle (GST_VIDEO_OVERLAY (appCtx
              [i]->pipeline.instance_bins[0].sink_bin.sub_bins[j].sink),
          (gulong) windows[i]);
      gst_video_overlay_expose (GST_VIDEO_OVERLAY (appCtx[i]->
              pipeline.instance_bins[0].sink_bin.sub_bins[j].sink));
      if (!x_event_thread)
        x_event_thread = g_thread_new ("nvds-window-event-thread",
            nvds_x_event_thread, NULL);
    }
#if !defined(__aarch64__)
    if (!prop.integrated) {
      if (gst_element_set_state (appCtx[i]->pipeline.pipeline,
              GST_STATE_PAUSED) == GST_STATE_CHANGE_FAILURE) {
        NVGSTDS_ERR_MSG_V ("Failed to set pipeline to PAUSED");
        return_value = -1;
        goto done;
      }
    }
#endif
  }

  /* Dont try to set playing state if error is observed */
  if (return_value != -1) {
    for (i = 0; i < num_instances; i++) {
      if (gst_element_set_state (appCtx[i]->pipeline.pipeline,
              GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {

        g_print ("\ncan't set pipeline to playing state.\n");
        return_value = -1;
        goto done;
      }
      if (appCtx[i]->config.pipeline_recreate_sec)
        g_timeout_add_seconds (appCtx[i]->config.pipeline_recreate_sec,
            recreate_pipeline_thread_func, appCtx[i]);
    }
  }

  print_runtime_commands ();

  changemode (1);

  g_timeout_add (40, event_thread_func, NULL);
  g_main_loop_run (main_loop);

  changemode (0);

done:

  g_print ("Quitting\n");
  for (i = 0; i < num_instances; i++) {
    if (appCtx[i]->return_value == -1)
      return_value = -1;
    destroy_pipeline (appCtx[i]);

    g_mutex_lock (&disp_lock);
    if (windows[i])
      XDestroyWindow (display, windows[i]);
    windows[i] = 0;
    g_mutex_unlock (&disp_lock);

    g_free (appCtx[i]);
  }

  g_mutex_lock (&disp_lock);
  if (display)
    XCloseDisplay (display);
  display = NULL;
  g_mutex_unlock (&disp_lock);
  g_mutex_clear (&disp_lock);

  if (main_loop) {
    g_main_loop_unref (main_loop);
  }

  if (ctx) {
    g_option_context_free (ctx);
  }

  if (return_value == 0) {
    g_print ("App run successful\n");
  } else {
    g_print ("App run failed\n");
  }

  gst_deinit ();

  return return_value;
}
