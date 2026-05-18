# 1. 把 GGUF 文件从电脑推送到手机临时目录
  
  adb push qwen3-0.6b.gguf /data/local/tmp/

# 2. 复制到 app 私有目录（app 才能读）

  adb shell "run-as org.osh26.llama cp /data/local/tmp/qwen3-0.6b.gguf /data/data/org.osh26.llama/files/models/"

# 或者直接推送到可读目录再用 app 加载

  adb push qwen3-0.6b.gguf /sdcard/Android/data/org.osh26.llama/files/models/
