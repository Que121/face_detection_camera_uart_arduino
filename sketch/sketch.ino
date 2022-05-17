void setup() {
  Serial.begin(115200);         // 启动串行连接，波特率为115200
}

void loop() {
  if(Serial.available() > 0){                         // 检查缓冲区内是否有可用的字节可供读取
    String str = Serial.readString();                 // 读取字节为字符串
    if(str.equals("1\n")){      
      Serial.println("Arduino: 1\n");
    }
  }
}
