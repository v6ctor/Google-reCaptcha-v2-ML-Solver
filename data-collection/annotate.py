from ultralytics.data.annotator import auto_annotate

auto_annotate(data="/images", det_model="yolo11n.pt", sam_model="mobile_sam.pt")