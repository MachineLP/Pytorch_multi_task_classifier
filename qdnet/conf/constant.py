import os
import threading

class Constant():

    RESNEST_LIST = ['resnest50', 'resnest101', 'resnest200', 'resnest269']
    RESNET_LIST = ['resnet18', 'resnet34', 'resnet50', 'tf_mobilenetv3_small_minimal_100'] 
    SERESNEXT_LIST = ['seresnext101']
    GEFFNET_LIST = ['GenEfficientNet', 'mnasnet_050', 'mnasnet_075', 'mnasnet_100', 'mnasnet_b1', 
    'mnasnet_140', 'semnasnet_050', 'semnasnet_075', 'semnasnet_100', 'mnasnet_a1', 'semnasnet_140', 
    'mnasnet_small','mobilenetv2_100', 'mobilenetv2_140', 'mobilenetv2_110d', 'mobilenetv2_120d', 'fbnetc_100', 
    'spnasnet_100', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',  'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 
    'efficientnet_b6', 'efficientnet_b7', 'efficientnet_b8', 'efficientnet_l2', 'efficientnet_es', 'efficientnet_em', 'efficientnet_el', 
    'efficientnet_cc_b0_4e', 'efficientnet_cc_b0_8e', 'efficientnet_cc_b1_8e', 'efficientnet_lite0', 'efficientnet_lite1', 'efficientnet_lite2', 
    'efficientnet_lite3', 'efficientnet_lite4', 'tf_efficientnet_b0', 'tf_efficientnet_b1', 'tf_efficientnet_b2', 'tf_efficientnet_b3', 
    'tf_efficientnet_b4', 'tf_efficientnet_b5', 'tf_efficientnet_b6', 'tf_efficientnet_b7', 'tf_efficientnet_b8', 'tf_efficientnet_b0_ap',
    'tf_efficientnet_b1_ap', 'tf_efficientnet_b2_ap', 'tf_efficientnet_b3_ap', 'tf_efficientnet_b4_ap', 'tf_efficientnet_b5_ap', 
    'tf_efficientnet_b6_ap', 'tf_efficientnet_b7_ap', 'tf_efficientnet_b8_ap', 'tf_efficientnet_b0_ns', 'tf_efficientnet_b1_ns', 
    'tf_efficientnet_b2_ns', 'tf_efficientnet_b3_ns', 'tf_efficientnet_b4_ns', 'tf_efficientnet_b5_ns', 'tf_efficientnet_b6_ns', 
    'tf_efficientnet_b7_ns', 'tf_efficientnet_l2_ns', 'tf_efficientnet_l2_ns_475', 'tf_efficientnet_es', 'tf_efficientnet_em', 
    'tf_efficientnet_el', 'tf_efficientnet_cc_b0_4e', 'tf_efficientnet_cc_b0_8e', 'tf_efficientnet_cc_b1_8e', 'tf_efficientnet_lite0', 
    'tf_efficientnet_lite1', 'tf_efficientnet_lite2', 'tf_efficientnet_lite3', 'tf_efficientnet_lite4', 'mixnet_s', 'mixnet_m', 'mixnet_l', 
    'mixnet_xl', 'tf_mixnet_s', 'tf_mixnet_m', 'tf_mixnet_l', 'mobilenetv3_rw', 
    'mobilenetv3_large_075', 'mobilenetv3_large_100', 'mobilenetv3_large_minimal_100',
    'mobilenetv3_small_075', 'mobilenetv3_small_100', 'mobilenetv3_small_minimal_100',
    'tf_mobilenetv3_large_075', 'tf_mobilenetv3_large_100', 'tf_mobilenetv3_large_minimal_100',
    'tf_mobilenetv3_small_075', 'tf_mobilenetv3_small_100', 'tf_mobilenetv3_small_minimal_100']


