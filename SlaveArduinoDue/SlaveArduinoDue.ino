/* Inisiasi LCD 20x4 I2C */
#include <LiquidCrystal_I2C.h>
LiquidCrystal_I2C lcd(0x27, 20, 4);

/* inisiasi Pin */
#define PWM 9 // PWM
#define HALLSEN_A 3
#define ENCODER_CONSTANT 2

//#define ADDRESS 0x04  // Alamat I2C Arduino
//#define DATA_SIZE 3   // Jumlah data yang akan diterima

int interval = 60;
int motorSpeed = 0;
long previousMillis = 0;
long currentMillis = 0;
float elapsedMillis = 0;
float dt = 0.01;

/* inisiasi Jenis Tipe Data */
int GT = 0;
int data = 0;
int rpmPID = 0;
int rpmMotor = 0;
int rpmMotorTacho = 0;
int rpmMotor2 = 0;
int rpmMotorReal = 0;
int motorPwm = 0;
int encoderValue = 0;
int error = 0;
int last_error = 0;
int SampleTime;
float integral;
float derivative;
unsigned long lastTime;
//double ITerm, lastInput;
int pot = A0;/* analog pin for potentiometer for LED brightness control*/
int Value = 0;/* declaring variable for storing the potentiometer v*/
const float pulsesPerRevolution = 32; // Replace with your encoder's PPR
const float gearRatio = 98.7;          // Replace with your gear ratio
const float wheelCircumference = 2.0 * 3.14159 * 0.17; // Replace with your wheel circumference (m)
//const float pulses = 8;

/* konstanta Nilai PID */
float kp = 0.047038;
float ki = 0.021396;
float kd = 0.015959;

/* set point  */
int sp = 0;
//int sp0 = 0;int sp60 = 567;int sp61 = 577;int sp62 = 587;
//int sp63 = 597;int sp64 = 607;int sp65 = 615;int sp66 = 634;
//int sp67 = 628;int sp68 = 652;int sp69 = 652;int sp70 = 666;
//int sp71 = 670;int sp72 = 690;int sp73 = 690;int sp74 = 709;
//int sp75 = 709;int sp76 = 721;int sp77 = 729;int sp78 = 739;
//int sp79 = 749;int sp80 = 755;int sp81 = 765;int sp82 = 775;
//int sp83 = 788;int sp84 = 794;int sp85 = 806;int sp86 = 824;
//int sp87 = 824;int sp88 = 832;int sp89 = 844;int sp90 = 863;
//int sp91 = 863;int sp92 = 871;int sp93 = 881;int sp94 = 889;
//int sp95 = 901;int sp96 = 909;int sp97 = 917;int sp98 = 927;
//int sp99 = 940;int sp100 = 946;

/* Data Dari Raspy */
int c;
int adjspeed;
int aktifasiLaneKeeping;

/*Pin Relay*/
const int PIN39 = 39;
const int PIN47 = 47;

//int receivedData[DATA_SIZE];  // Array untuk menyimpan data yang diterima

void setup()
{
  Serial.begin(9600);
  //Serial1.begin(115200);
  //Serial1.begin(115200);
  lcd.init();
  lcd.backlight();
  //tabel daq
  //Serial1.println("CLEARDATA");        //This string is defined as a
  // commmand for the Excel VBA
  // to clear all the rows and columns
  //Serial.println("LABEL,Computer Time,Time (Sec.),Kelas,AdjSpeed,Kecepatan (RPM),Kecepatan (km/jam)");
  //Serial1.println("LABEL,Computer Time,Time (Sec.),Kecepatan (RPM),Kecepatan (km/jam)");
  pinMode(PIN39, OUTPUT);
  pinMode(PIN47, OUTPUT);
  pinMode(HALLSEN_A , INPUT_PULLUP);
  pinMode(ENCODER_CONSTANT , INPUT_PULLUP);
  pinMode(PWM, OUTPUT);
  analogWrite(PWM, 0);

  // encoderValue = 0;
  attachInterrupt(digitalPinToInterrupt(HALLSEN_A), updateEncoder, RISING);
  previousMillis = millis();
}

void loop() {
  while ((data <= 100000))
  {
    TerimaData();
    Value = analogRead(pot);/* getting the value of potentiometer*/
    adjspeed = map(Value, 0, 1023, 0, 120); /* scalarizing the analog values in the  range of 0 to 100*/
    currentMillis = millis();
    if (currentMillis - previousMillis > interval)
    {
      if (c == 2 && GT == 0 && aktifasiLaneKeeping == 0 && adjspeed <= 60)
      {
        sp = 0;
        lcd.setCursor(0, 1);
        lcd.print("OFF                           ");
      }
      else if (c == 2 && GT == 0 && aktifasiLaneKeeping == 1 && adjspeed <= 60)
      {
        GT = 1;
        sp = 572;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 2 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed <= 60)
      {
        digitalWrite(PIN39, HIGH);
        digitalWrite(PIN47, HIGH);
        sp = 572;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 2 && GT == 1 && aktifasiLaneKeeping == 0 && adjspeed <= 60)
      {
        GT = 0;
        sp = 0;
        digitalWrite(PIN39, LOW);
        digitalWrite(PIN47, LOW);
        //Serial.print("MA");
        lcd.setCursor(0, 1);
        lcd.print("OFF                           ");
      }
      else if (c == 1 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 0)
      {
        sp = 762
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 1 && GT == 1 && aktifasiLaneKeeping == 1 && 0 < adjspeed < 200)
      {
        sp = 762
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 1 && GT == 0 && aktifasiLaneKeeping == 1 )
      {
        sp = 0;
        //Serial.print("menunggu aktifasi");
        lcd.setCursor(0, 1);
        lcd.print("OFF                            ");
      }
      else if (c == 1 && GT == 1 && aktifasiLaneKeeping == 0)
      {
        sp = 0;
        //Serial.print("menunggu aktifasi");
        lcd.setCursor(0, 1);
        lcd.print("OFF                            ");
      }
      else if (c == 1 && GT == 0 && aktifasiLaneKeeping == 0)
      {
        sp = 0;
        //Serial.print("menunggu aktifasi");
        lcd.setCursor(0, 1);
        lcd.print("OFF                            ");
      }
      else if (c == 3 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 0)
      {
        sp = 955;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 3 && GT == 1 && aktifasiLaneKeeping == 1 && 0 < adjspeed < 200)
      {
        sp = 955;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 3 && GT == 1 && aktifasiLaneKeeping == 0)
      {
        sp = 0;
        //Serial.print("menunggu aktifasi");
        lcd.setCursor(0, 1);
        lcd.print("OFF                            ");
      }
      else if (c == 3 && GT == 0 && aktifasiLaneKeeping == 1)
      {
        sp = 0;
        //Serial.print("menunggu aktifasi");
        lcd.setCursor(0, 1);
        lcd.print("OFF                            ");
      }
      else if (c == 3 && GT == 0 && aktifasiLaneKeeping == 0)
      {
        sp = 0;
        //Serial.print("menunggu aktifasi");
        lcd.setCursor(0, 1);
        lcd.print("OFF                            ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed <= 60)
      {
        sp = 572;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 61)
      {
        sp = 582;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 62)
      {
        sp = 592;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 63)
      {
        sp = 602;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 64)
      {
        sp = 613;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 65)
      {
        sp = 621;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 66)
      {
        sp = 632;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 67)
      {
        sp = 640;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 68)
      {
        sp = 650;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 69)
      {
        sp = 658;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 70)
      {
        sp = 672;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 71)
      {
        sp = 676;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 72)
      {
        sp = 688;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 73)
      {
        sp = 696;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 74)
      {
        sp = 707;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 75)
      {
        sp = 716;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 76)
      {
        sp = 728;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 77)
      {
        sp = 736;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 78)
      {
        sp = 746;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 79)
      {
        sp = 756;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 80)
      {
        sp = 762
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 81)
      {
        sp = 812;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 82)
      {
        sp = 782;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 83)
      {
        sp = 795;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 84)
      {
        sp = 801;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 85)
      {
        sp = 814;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 86)
      {
        sp = 822;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 87)
      {
        sp = 832;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 88)
      {
        sp = 840;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 89)
      {
        sp = 852;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 90)
      {
        sp = 860;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 91)
      {
        sp = 871;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 92)
      {
        sp = 879;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 93)
      {
        sp = 889;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 94)
      {
        sp = 897;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 95)
      {
        sp = 910;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 96)
      {
        sp = 918;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 97)
      {
        sp = 926;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 98)
      {
        sp = 936;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 99)
      {
        sp = 949;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed >= 100)
      {
        sp = 955;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 4 && GT == 1 && aktifasiLaneKeeping == 0)
      {
        sp = 0;
        //Serial.print("menunggu aktifasi");
        lcd.setCursor(0, 1);
        lcd.print("OFF                            ");
      }
      else if (c == 4 && GT == 0 && aktifasiLaneKeeping == 1)
      {
        sp = 0;
        //Serial.print("menunggu aktifasi");
        lcd.setCursor(0, 1);
        lcd.print("OFF                            ");
      }
      else if (c == 0 && GT == 0 && aktifasiLaneKeeping == 0)
      {
        sp = 0;
        //Serial.print("menunggu aktifasi");
        lcd.setCursor(0, 1);
        lcd.print("OFF                            ");
      }
      else if (c == 0 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed <= 60)
      {
        sp = 572;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 0 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 61)
      {
        sp = 582;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 0 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 62)
      {
        sp = 592;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 0 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 63)
      {
        sp = 602;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 0 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 64)
      {
        sp = 613;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 0 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 65)
      {
        sp = 621;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 0 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 66)
      {
        sp = 632;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 0 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 67)
      {
        sp = 640;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 0 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 68)
      {
        sp = 650;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 0 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 69)
      {
        sp = 658;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 0 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 70)
      {
        sp = 672;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 0 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 71)
      {
        sp = 676;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 0 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 72)
      {
        sp = 688;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 0 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 73)
      {
        sp = 696;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 0 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 74)
      {
        sp = 797;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 0 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 75)
      {
        sp = 716;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 0 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 76)
      {
        sp = 728;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 0 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 77)
      {
        sp = 736;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 0 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 78)
      {
        sp = 746;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 0 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 79)
      {
        sp = 756;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 0 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed >= 80)
      {
        sp = 762
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 0 && GT == 1 && aktifasiLaneKeeping == 0)
      {
        sp = 0;
        //Serial.print("menunggu aktifasi");
        lcd.setCursor(0, 1);
        lcd.print("OFF                            ");
      }
      else if (c == 0 && GT == 0 && aktifasiLaneKeeping == 1)
      {
        sp = 0;
        //Serial.print("menunggu aktifasi");
        lcd.setCursor(0, 1);
        lcd.print("OFF                            ");
      }
      else if (c == 0 && GT == 0 && aktifasiLaneKeeping == 0)
      {
        sp = 0;
        //Serial.print("menunggu aktifasi");
        lcd.setCursor(0, 1);
        lcd.print("OFF                            ");
      }
      else if (c == 5 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed <= 60)
      {
        sp = 572;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 5 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 61)
      {
        sp = 582;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 5 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 62)
      {
        sp = 592;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 5 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 63)
      {
        sp = 602;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 5 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 64)
      {
        sp = 613;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 5 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 65)
      {
        sp = 621;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 5 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 66)
      {
        sp = 632;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 5 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 67)
      {
        sp = 640;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 5 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 68)
      {
        sp = 650;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 5 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 69)
      {
        sp = 658;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 5 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 70)
      {
        sp = 672;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 5 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 71)
      {
        sp = 676;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 5 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 72)
      {
        sp = 688;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 5 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 73)
      {
        sp = 696;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 5 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 74)
      {
        sp = 707;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 5 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 75)
      {
        sp = 716;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 5 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 76)
      {
        sp = 728;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 5 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 77)
      {
        sp = 736;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 5 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 78)
      {
        sp = 746;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 5 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed == 79)
      {
        sp = 756;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 5 && GT == 1 && aktifasiLaneKeeping == 1 && adjspeed >= 80)
      {
        sp = 762
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 5 && GT == 1 && aktifasiLaneKeeping == 0)
      {
        sp = 0;
        //Serial.print("menunggu aktifasi");
        lcd.setCursor(0, 1);
        lcd.print("OFF                            ");
      }
      else if (c == 5 && GT == 0 && aktifasiLaneKeeping == 1)
      {
        sp = 0;
        //Serial.print("menunggu aktifasi");
        lcd.setCursor(0, 1);
        lcd.print("OFF                            ");
      }
      else if (c == 5 && GT == 0 && aktifasiLaneKeeping == 0)
      {
        sp = 0;
        //Serial.print("menunggu aktifasi");
        lcd.setCursor(0, 1);
        lcd.print("OFF                            ");
      }
      else if (c == 6 && GT == 1 && aktifasiLaneKeeping == 1)
      {
        sp = 572;
        lcd.setCursor(0, 1);
        lcd.print("ON                           ");
      }
      else if (c == 6 && GT == 1 && aktifasiLaneKeeping == 0)
      {
        sp = 0;
        //Serial.print("menunggu aktifasi");
        lcd.setCursor(0, 1);
        lcd.print("OFF                            ");
      }
      else if (c == 6 && GT == 0 && aktifasiLaneKeeping == 1)
      {
        sp = 0;
        //Serial.print("menunggu aktifasi");
        lcd.setCursor(0, 1);
        lcd.print("OFF                            ");
      }
      else if (c == 6 && GT == 0 && aktifasiLaneKeeping == 0)
      {
        sp = 0;
        //Serial.print("menunggu aktifasi");
        lcd.setCursor(0, 1);
        lcd.print("OFF                            ");
      }
      else
      {
        sp = 0;
        //Serial.print("menunggu aktifasi");
        lcd.setCursor(0, 1);
        lcd.print("OFF                            ");
      }

      /* Hitung RPM */
      //rpmPID = (float)(encoderValue * 2.73); // Kalkulasi kecepatan motor DC (RPM)
      rpmMotor = ((float)(encoderValue) * 60.000 / pulsesPerRevolution);
      rpmMotorTacho = ((float)(encoderValue) * 60.000 / pulsesPerRevolution) * 16;
      float realRPM = ((float)(encoderValue) * 60.000 / pulsesPerRevolution) * 16;
      //int realRPM = (0.0305 * rpmMotorTacho) - 0.8462;
      int outputRPM = realRPM / gearRatio;
      float speed_kmph = (outputRPM * wheelCircumference * 60.0) / 1000.0;
      //rpmMotor = ((float)(encoderValue) * 60.000) / (pulses / interval);
      //rpmMotor2 = (float)(encoderValue) * 60.000 / pulsesPerRevolution ;
      //rpmMotorReal = (4.1794 * rpmMotor) - 484.22;
      previousMillis = currentMillis;

      /* Menampilkan Data Pada Serial Monitor */
      //      Serial.print(sp);
      //      Serial.print(",");
      //      //Serial.print(rpmPID);
      //      //Serial.print(",");
      //      Serial.print(rpmMotor);
      //      Serial.print(",");
      Serial.print(rpmMotorTacho);
      //      Serial.print(",");

      //      Serial.print(realRPM);
      //      Serial.print(",");
      //      Serial1.print(outputRPM);
      //      Serial1.print(",");
      //      Serial1.println(speed_kmph);
      //KirimData();
      //      Serial.print(",");
      //      Serial.println(motorSpeed);
      /*Rumus PID */
      error = sp - rpmMotor; // error
      integral += error; // integral
      derivative = (error - last_error); // Derivative
      motorSpeed = ((kp * error) + (ki * integral) + (kd * (derivative))); //Penambahan Rumus PID

      /* Kontrol Motor DC */
      if (motorSpeed > 255) motorSpeed = 255;
      else if (motorSpeed < 0) motorSpeed = 0;
      analogWrite(PWM, motorSpeed);
      last_error = error;
      encoderValue = 0;
      data++;

      kirimData();
      lcd.setCursor(0, 0);
      lcd.print("c:");
      lcd.setCursor(2, 0);
      lcd.print(c);
      lcd.setCursor(4, 0);
      lcd.print("adj:");
      lcd.setCursor(8, 0);
      lcd.print(adjspeed);
      lcd.setCursor(11, 0);
      lcd.print("ALK:");
      lcd.setCursor(15, 0);
      lcd.print(aktifasiLaneKeeping);
      lcd.setCursor(17, 0);
      lcd.print("G:");
      lcd.setCursor(19, 0);
      lcd.print(GT);
      lcd.setCursor(5, 1);
      lcd.print("rpm:");
      lcd.setCursor(9, 1);
      lcd.print(rpmMotorTacho);
      lcd.setCursor(0, 2);
      lcd.print("rpmreal:");
      lcd.setCursor(8, 2);
      lcd.print(outputRPM);
      lcd.setCursor(0, 3);
      lcd.print("kec:");
      lcd.setCursor(4, 3);
      lcd.print(speed_kmph);
      lcd.setCursor(8, 3);
      lcd.print("km/jam");
      //Serial.println(speed_kmph);

      //PLX DAq =======================================================
      //      Serial.print("DATA,TIME,");// computer time
      //      Serial.print(currentMillis / 1000); // menampilkan waktu sampling
      //      Serial.print(",");
      //Serial.println(rpmMotorTacho);//menampilkan kecepatan (rpm)
      //      Serial.print(",");
      //Serial.println(speed_kmph);//menampilkan kecepatan (km/jam)
      //delay(5);

      //      Wire.beginTransmission(8); // Alamat slave Arduino (misalnya, 8)
      //      Wire.write((byte*)&rpmMotorTacho, sizeof(rpmMotorTacho)); // Mengirim data integer
      //      //Wire.endTransmission();
    }
    analogWrite(PWM, motorSpeed);
  }
  analogWrite(PWM, 0);
}

/* Void perintah untuk Menambah Nilai Encoder */
void updateEncoder()
{
  encoderValue++;
}

void TerimaData()
{
  if (Serial.available() > 0)
  {
    // Membaca data dari koneksi serial
    c = Serial.parseInt();  // Membaca nilai c
    Serial.read();  // Membaca dan mengabaikan koma (,)
    //    adjspeed = Serial.parseInt();  // Membaca nilai adjspeed
    //    Serial.read();  // Membaca dan mengabaikan koma (,)
    aktifasiLaneKeeping = Serial.parseInt();  // Membaca nilai aktifasiLaneKeeping
  }
  //  rpmMotor = ((float)(encoderValue) * 60.000 / pulsesPerRevolution);
  //  rpmMotorTacho = ((float)(encoderValue) * 60.000 / pulsesPerRevolution) * 33;
  //  float realRPM = ((float)(encoderValue) * 60.000 / pulsesPerRevolution) * 33;
  //  //int realRPM = (0.0305 * rpmMotorTacho) - 0.8462;
  //  float outputRPM = realRPM / gearRatio;
  //  float speed_kmph = (outputRPM * wheelCircumference * 60.0) / 1000.0;
  //  Serial.print(rpmMotorTacho);
  //  Serial.print(",");
  //  Serial.print(speed_kmph);
}

void kirimData()
{
  Wire.beginTransmission(8); // Mulai transmisi ke alamat slave (8)
  Wire.write((byte*)&c, sizeof(c)); // Kirim data1
  Wire.write((byte*)&adjspeed, sizeof(adjspeed)); // Kirim data2
  Wire.write((byte*)&sp, sizeof(sp)); // Kirim data3
  Wire.write((byte*)&rpmMotorTacho, sizeof(rpmMotorTacho)); // Kirim data3
  //Wire.write((byte*)&sp, sizeof(sp)); // Kirim data3
  Wire.endTransmission(); // Selesai transmisi
  //delay(5);
}
