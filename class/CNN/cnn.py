import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Rescaling, Dropout
import pathlib
import os

# 1. กำหนดที่อยู่ไฟล์
data_dir_path = r"C:\Users\CPE-CDTI_X390\Desktop\AI\class\CNN\dataset"
data_dir = pathlib.Path(data_dir_path)

# --- [ส่วนที่แก้เพิ่ม 1] เช็คว่าโฟลเดอร์มีจริงไหม ---
if not data_dir.exists():
    print(f"❌ Error: ไม่เจอโฟลเดอร์ที่ {data_dir_path}")
    print("กรุณาเช็ค path อีกครั้ง หรือเช็คว่าสร้างโฟลเดอร์ไว้ถูกที่ไหม")
    exit() # หยุดทำงานทันทีถ้าไม่เจอโฟลเดอร์

# กำหนดขนาดรูปภาพ
img_height = 64
img_width = 64
batch_size = 32

print(f"กำลังโหลดข้อมูลจาก: {data_dir}")

# 2. โหลดข้อมูล (Load Data)
# หมายเหตุ: ถ้าข้อมูลน้อยมาก (เช่นมีแค่ 2-10 รูป) ผมแนะนำให้ปิด validation_split ชั่วคราว
# แต่ถ้ามีรูปเยอะ (หลักร้อย) ให้เปิด validation_split=0.2 ไว้เหมือนเดิม

try:
    train_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size
    )
except ValueError as e:
    print("\n⚠️ เกิดปัญหาตอนโหลดข้อมูล!")
    print("สาเหตุที่เป็นไปได้: คุณมีรูปน้อยเกินไปจนแบ่ง 80/20 ไม่ได้ หรือ คุณลืมสร้างโฟลเดอร์ย่อย (Class)")
    print(f"Error details: {e}")
    exit()

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\n✅ เจอทั้งหมด {num_classes} คลาส ได้แก่: {class_names}")

# 3. สร้างโมเดล
model = Sequential([
  Input(shape=(img_height, img_width, 3)),
  Rescaling(1./255),
  
  # Layer 1
  Conv2D(32, 3, padding='same', activation='relu'),
  BatchNormalization(),
  MaxPool2D(),
  
  # Layer 2
  Conv2D(64, 3, padding='same', activation='relu'),
  BatchNormalization(),
  MaxPool2D(),
  
  # Layer 3 (เพิ่มมาเพื่อให้เรียนรู้เส้นโค้งซับซ้อนได้ดีขึ้น)
  Conv2D(128, 3, padding='same', activation='relu'),
  BatchNormalization(),
  MaxPool2D(),
  
  Flatten(),
  
  # --- [ส่วนที่แก้เพิ่ม 2] ใส่ Dropout ---
  Dense(128, activation='relu'),
  Dropout(0.5), # สุ่มลืมข้อมูล 50% เพื่อป้องกันการจำข้อสอบ (สำคัญมากถ้าข้อมูลน้อย)
  
  Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary()

# 4. สั่งเทรน
print("\n🚀 เริ่มเทรนโมเดล...")
# ถ้าข้อมูลน้อย ให้เพิ่ม epochs เยอะหน่อย (เช่น 10-20 รอบ)
epochs = 10 
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

print("\n🎉 เทรนเสร็จเรียบร้อย!")