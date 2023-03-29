
import os
import cv2


video_path = "./data/img/"
all_labels_path = os.listdir(video_path)
black_list3 = []

for per_label_path in all_labels_path:
    if 'csv' in per_label_path:
        continue
    per_all_img_path = os.path.join(video_path, per_label_path)
    all_image_path = os.listdir(per_all_img_path)
    for per_image_path in all_image_path:
        per_img_path = os.path.join(video_path, per_label_path, per_image_path)
        try:
            img = cv2.imread( per_img_path )
            if img is None:
                black_list3.append( per_image_path )
                # os.system( "rm -r '{}'".format(per_img_path) )
                print (222)
        except:
            black_list3.append( per_image_path )
            # os.system( "rm -r {}".format(per_img_path) )
            print (222)
            continue
        
print ( "Finish!!" )
print ( list(set( black_list3 )) )
