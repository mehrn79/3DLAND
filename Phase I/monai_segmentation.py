from monai.bundle import ConfigParser

config_path = "monai_wholeBody_ct_segmentation/configs/inference.json"

parser = ConfigParser()
parser.read_config(config_path)

evaluator = parser.get_parsed_content("evaluator")

try:
    evaluator.run()
except Exception as e:
    print(f"⚠️ خطا در اجرای evaluator: {e}")
