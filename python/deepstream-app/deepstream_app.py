#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys
sys.path.append('../')
import gi
import argparse
import configparser
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GLib, Gst, GstRtspServer
from ctypes import *
import time
import sys
import math
import platform
from common.platform_info import PlatformInfo
from common.bus_call import bus_call
from common.FPS import PERF_DATA

import pyds

import paho.mqtt.client as mqtt
import json
import csv

mqttc = None

perf_data = None
sources_path = None
output_type = None
codec = None
bitrate = 4000000

MAX_DISPLAY_LEN=64
MUXER_OUTPUT_WIDTH=1920
MUXER_OUTPUT_HEIGHT=1080
MUXER_BATCH_TIMEOUT_USEC = 33000
TILED_OUTPUT_WIDTH=1920
TILED_OUTPUT_HEIGHT=1000
GST_CAPS_FEATURES_NVMM="memory:NVMM"
OSD_PROCESS_MODE= 0
OSD_DISPLAY_TEXT= 1

count_offset = {}
count_current = {}

def calculate_current_count(stream_index = None):
    global count_current
    global count_offset
    data = {}
    for stream_id in count_current:
        if stream_id in count_offset:
            data[stream_id] = {}
            for lc_id in count_current[stream_id]:
                if lc_id in count_offset[stream_id]:
                    data[stream_id][lc_id] = count_current[stream_id][lc_id] - count_offset[stream_id][lc_id]
                else:
                    data[stream_id][lc_id] = count_current[stream_id][lc_id]
        else:
            data[stream_id] = count_current[stream_id]
    if stream_index is None:
        return data
    else:
        if stream_index in data:
            return data[stream_index]
        else:
            return {}
    

# nvanlytics_src_pad_buffer_probe  will extract metadata received on nvtiler sink pad
# and update params for drawing rectangle, object information etc.
def nvanalytics_src_pad_buffer_probe(pad,info,u_data):
    global count_current
    frame_number=0
    num_rects=0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        l_obj=frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        #print("#"*50)
    
        # Get meta data from NvDsAnalyticsFrameMeta
        l_user = frame_meta.frame_user_meta_list
        while l_user:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSFRAME.USER_META"):
                    user_meta_data = pyds.NvDsAnalyticsFrameMeta.cast(user_meta.user_meta_data)
                    if user_meta_data.objInROIcnt: print("Objs in ROI: {0}".format(user_meta_data.objInROIcnt))                    
                    if user_meta_data.objLCCumCnt: 
                        #print("Linecrossing Cumulative: {0}".format(user_meta_data.objLCCumCnt))
                        num_rects = user_meta_data.objLCCumCnt
                        count_current[frame_meta.pad_index] = user_meta_data.objLCCumCnt
                    #if user_meta_data.objLCCurrCnt: print("Linecrossing Current Frame: {0}".format(user_meta_data.objLCCurrCnt))
                    if user_meta_data.ocStatus: print("Overcrowding status: {0}".format(user_meta_data.ocStatus))
            except StopIteration:
                break
            try:
                l_user = l_user.next
            except StopIteration:
                break
        
        #print("Frame Number=", frame_number, "stream id=", frame_meta.pad_index, "Number of Objects=",num_rects,"Vehicle_count=",obj_counter[PGIE_CLASS_ID_VEHICLE],"Person_count=",obj_counter[PGIE_CLASS_ID_PERSON])
        
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        
        data = calculate_current_count(frame_meta.pad_index)
        
        py_nvosd_text_params.display_text = "{}".format(data)

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 35
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        
        # Update frame rate through this probe
        stream_index = "stream{0}".format(frame_meta.pad_index)
        global perf_data
        perf_data.update_fps(stream_index)
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
        #print("#"*50)

    return Gst.PadProbeReturn.OK

def cb_newpad(decodebin, decoder_src_pad,data):
    print("In cb_newpad\n")
    caps=decoder_src_pad.get_current_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=",gstname)
    if(gstname.find("video")!=-1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=",features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy,Object,name,user_data):
    print("Decodebin child added:", name, "\n")
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)

def create_source_bin(index,uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name="source-bin-%02d" %index
    print(bin_name)
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri",uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected with result code {reason_code}")
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("command")
    #client.subscribe("chakkarat/status_reset")

def on_message(client, userdata, msg):
    global count_offset
    global count_current
    
    try:
        data = json.loads(msg.payload)	            
    except:
        print("Error on json parsing")
        return None

    if msg.topic == "command":
        if "command" in data and "stream_id" in data:
            if data["command"] == "reset":
                stream_id = data["stream_id"]
                if stream_id in count_current:
                    count_offset[stream_id] = count_current[stream_id]
                    print("Reset stream :", stream_id)
                else:
                    print("No stream :", stream_id)
        else:
            print("Incorrect command")
    if msg.topic == "chakkarat/status_reset":
        if "value" in data:
            if data["value"] == "True":
                for stream_id in count_current:
                    count_offset[stream_id] = count_current[stream_id]
                print("Reset all stream")
		            
    
def sent_mqtt():
    global mqttc
    topic = "egg_count"
    data = calculate_current_count()
    data_str = json.dumps(data)
    mqttc.publish(topic, data_str)
    print("Sent Data to ", topic, "=>", data_str)
    return True

def parse_source(filename):
    csvfile = open(filename, newline='')
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    uri_list = []
    for row in reader:
        if row[0] == "1":
            uri_list.append(row[2])
    return uri_list

def main(args):
    # Check input arguments
    #if len(args) < 2:
    #    sys.stderr.write("usage: %s <uri1> [uri2] ... [uriN]\n" % args[0])
    #    sys.exit(1)
    global sources_path
    uri_list = parse_source(sources_path)

    global perf_data
    global mqttc
    perf_data = PERF_DATA(len(uri_list))
    number_sources=len(uri_list)

    platform_info = PlatformInfo()
    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin ",i," \n ")
        uri_name=uri_list[i]
        if uri_name.find("rtsp://") == 0 :
            is_live = True
        source_bin=create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname="sink_%u" %i
        sinkpad= streammux.request_pad_simple(padname) 
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad=source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
    queue1=Gst.ElementFactory.make("queue","queue1")
    queue2=Gst.ElementFactory.make("queue","queue2")
    queue3=Gst.ElementFactory.make("queue","queue3")
    queue4=Gst.ElementFactory.make("queue","queue4")
    queue5=Gst.ElementFactory.make("queue","queue5")
    queue6=Gst.ElementFactory.make("queue","queue6")
    queue7=Gst.ElementFactory.make("queue","queue7")
    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    pipeline.add(queue5)
    pipeline.add(queue6)
    pipeline.add(queue7)

    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    print("Creating nvtracker \n ")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")

    print("Creating nvdsanalytics \n ")
    nvanalytics = Gst.ElementFactory.make("nvdsanalytics", "analytics")
    if not nvanalytics:
        sys.stderr.write(" Unable to create nvanalytics \n")
    nvanalytics.set_property("config-file", "configs/config_nvdsanalytics.txt")

    print("Creating tiler \n ")
    tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")

    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    nvosd.set_property('process-mode',OSD_PROCESS_MODE)
    nvosd.set_property('display-text',OSD_DISPLAY_TEXT)
    #nvosd.set_property('display-clock',1)

    if output_type == "rtsp":
        nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")
        if not nvvidconv_postosd:
            sys.stderr.write(" Unable to create nvvidconv_postosd \n")
    
        # Create a caps filter
        caps = Gst.ElementFactory.make("capsfilter", "filter")
        caps.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
        )
        
        # Make the encoder
        if codec == "H264":
            encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
            print("Creating H264 Encoder")
        elif codec == "H265":
            encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
            print("Creating H265 Encoder")
        if not encoder:
            sys.stderr.write(" Unable to create encoder")
        encoder.set_property("bitrate", bitrate)
        if platform_info.is_integrated_gpu():
            encoder.set_property("preset-level", 1)
            encoder.set_property("insert-sps-pps", 1)
            #encoder.set_property("bufapi-version", 1)

        # Make the payload-encode video into RTP packets
        if codec == "H264":
            rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
            print("Creating H264 rtppay")
        elif codec == "H265":
            rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")
            print("Creating H265 rtppay")
        if not rtppay:
            sys.stderr.write(" Unable to create rtppay")

        # Make the UDP sink
        updsink_port_num = 5400
        sink = Gst.ElementFactory.make("udpsink", "udpsink")
        if not sink:
            sys.stderr.write(" Unable to create udpsink")

        sink.set_property("host", "224.224.255.255")
        sink.set_property("port", updsink_port_num)
        sink.set_property("async", False)
        sink.set_property("sync", 1)

    else:
        if platform_info.is_integrated_gpu():
            print("Creating nv3dsink \n")
            sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
            if not sink:
                sys.stderr.write(" Unable to create nv3dsink \n")
        else:
            if platform_info.is_platform_aarch64():
                print("Creating nv3dsink \n")
                sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
            else:
                print("Creating EGLSink \n")
                sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
            if not sink:
                sys.stderr.write(" Unable to create egl sink \n")

    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property('live-source', 1)

    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)
    pgie.set_property('config-file-path', "configs/config_pgie.txt")
    pgie_batch_size=pgie.get_property("batch-size")
    if(pgie_batch_size != number_sources):
        print("WARNING: Overriding infer-config batch-size",pgie_batch_size," with number of sources ", number_sources," \n")
        pgie.set_property("batch-size",number_sources)
    tiler_rows=int(math.sqrt(number_sources))
    tiler_columns=int(math.ceil((1.0*number_sources)/tiler_rows))
    tiler.set_property("rows",tiler_rows)
    tiler.set_property("columns",tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    sink.set_property("qos",0)

    #Set properties of tracker
    config = configparser.ConfigParser()
    config.read('configs/config_tracker.txt')
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
    tracker.set_property('display-tracking-id', 0)
    
    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(nvanalytics)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    if output_type == "rtsp":
        pipeline.add(nvvidconv_postosd)
        pipeline.add(caps)
        pipeline.add(encoder)
        pipeline.add(rtppay)
    pipeline.add(sink)

    # We link elements in the following order:
    # sourcebin -> streammux -> nvinfer -> nvtracker -> nvdsanalytics ->
    # nvtiler -> nvvideoconvert -> nvdsosd -> sink
    print("Linking elements in the Pipeline \n")
    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(queue2)
    queue2.link(tracker)
    tracker.link(queue3)
    queue3.link(nvanalytics)
    nvanalytics.link(queue4)
    queue4.link(tiler)
    tiler.link(queue5)
    queue5.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(queue6)
    if output_type == "rtsp":
        queue6.link(nvvidconv_postosd)
        nvvidconv_postosd.link(caps)
        caps.link(encoder)
        encoder.link(rtppay)
        rtppay.link(sink)
    else:
        queue6.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)
    nvanalytics_src_pad=nvanalytics.get_static_pad("src")
    if not nvanalytics_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        nvanalytics_src_pad.add_probe(Gst.PadProbeType.BUFFER, nvanalytics_src_pad_buffer_probe, 0)
        # perf callback function to print fps every 5 sec
        GLib.timeout_add(5000, perf_data.perf_print_callback)
        GLib.timeout_add(1000, sent_mqtt)

    # List the sources
    print("Now playing...")
    for i, source in enumerate(uri_list):
        print(i, ": ", source)
            
    # Start MQTT client for publishing data
    mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqttc.on_connect = on_connect
    mqttc.on_message = on_message

    mqttc.connect_async("localhost", 1883, 60)
    mqttc.loop_start()
    
    if output_type == "rtsp":
        # Start RTSP Server streaming
        rtsp_port_num = 8554

        server = GstRtspServer.RTSPServer.new()
        server.props.service = "%d" % rtsp_port_num
        server.attach(None)

        factory = GstRtspServer.RTSPMediaFactory.new()
        factory.set_launch(
            '( udpsrc name=pay0 port=%d buffer-size=524288 caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 " )'
            % (updsink_port_num, codec)
        )
        factory.set_shared(True)
        server.get_mount_points().add_factory("/ds-test", factory)

        print(
            "\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n"
            % rtsp_port_num
        )

    print("Starting pipeline \n")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
        
    # cleanup
    mqttc.loop_stop()
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)

def parse_args():
    parser = argparse.ArgumentParser(description='Application Help')
    parser.add_argument("-i", "--input", default="sources.csv",
                  help="Path to sources.csv")
    parser.add_argument("-o", "--output", default="rtsp",
                  help="Output type (rtsp,display), default=rtsp", choices=['rtsp','display'])
    parser.add_argument("-c", "--codec", default="H265",
                  help="RTSP Streaming Codec H264/H265 , default=H265", choices=['H264','H265'])

    args = parser.parse_args()
    global sources_path
    global output_type
    global codec
    sources_path = args.input
    output_type = args.output
    codec = args.codec

if __name__ == '__main__':
    parse_args()
    sys.exit(main(sys.argv))
