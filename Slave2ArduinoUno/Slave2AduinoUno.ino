#include <Wire.h>

int a, b, c, d; // Deklarasi variabel a, b, dan c
//double ITerm, lastInput;
const float pulsesPerRevolution = 32; // Replace with your encoder's PPR
const float gearRatio = 98.7;          // Replace with your gear ratio
const float wheelCircumference = 2.0 * 3.14159 * 0.17; // Replace with your wheel circumference (m)

void setup() {
  Wire.begin(8); // Inisialisasi I2C dengan alamat slave (8)
  Wire.onReceive(receiveEvent); // Mendaftarkan fungsi receiveEvent sebagai penanganan data masuk
  Serial.begin(9600); // Inisialisasi Serial Monitor
  Serial.println("CLEARDATA");        //This string is defined as a
  // commmand for the Excel VBA
  // to clear all the rows and columns
  Serial.println("LABEL,Computer Time,c,adjspeed,sp,rpm,rpmroda,km/jam");
}

void loop() {
  // Tidak ada yang perlu dilakukan pada loop di sisi penerima
}

void receiveEvent(int numBytes) {
  if (numBytes >= 8 * sizeof(int)) {
    int receivedData[8]; // Array untuk menyimpan data yang diterima

    Wire.readBytes((byte*)&receivedData, 8 * sizeof(int)); // Baca data yang diterima

    // Mengisi variabel a, b, dan c dengan nilai dari array receivedData
    a = receivedData[0];
    b = receivedData[2];
    c = receivedData[4];
    d = receivedData[6];
    //e = receivedData[8];

    int outputRPM = d / gearRatio;
    float speed_kmph = (outputRPM * wheelCircumference * 60.0) / 1000.0;
    //    Serial.println("Menerima data:");
    //    Serial.print("Data a: ");
    Serial.print("DATA,TIME,");// computer time
    Serial.print(a);
    Serial.print(",");
    //    Serial.print("Data b: ");
    Serial.print(b);
    Serial.print(",");
    //    Serial.print("Data c: ");
    Serial.print(c);
    Serial.print(",");
    //    Serial.print("Data d: ");
    Serial.print(d);
    Serial.print(",");
    //Serial.println(e);
    Serial.print(outputRPM);
    Serial.print(",");
    Serial.println(speed_kmph);
    delay(500);
  }
}
