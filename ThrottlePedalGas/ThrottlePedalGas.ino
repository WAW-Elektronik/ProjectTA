int pot = A0;/* analog pin for potentiometer for LED brightness control*/
int led = 6;/* defining the LED pin for Arduino */
int Value = 0;/* declaring variable for storing the potentiometer value*/
int LEDvalue = 0; /* variable that will store the scalarized value of pot*/
void setup() {
  Serial.begin(9600);
  pinMode(led, OUTPUT); /* defining the output mode for LED*/
  Serial.println("CLEARDATA");        //This string is defined as a
  // commmand for the Excel VBA
  // to clear all the rows and columns
  Serial.println("LABEL,Computer Time,c,adjspeed,alk,rpm");
}
void loop() {

  Value = analogRead(pot);/* getting the value of potentiometer*/        
  LEDvalue=map(Value, 0, 1023, 0, 200); /* scalarizing the analog values in the  range of 0 to 100*/
  analogWrite(led, LEDvalue); /* assigning the scalarized values to the LED */
  Serial.print("DATA,TIME,");// computer time
  //Serial.print("unmapped value :");
  Serial.print (Value);// printing the POT values in the serial monitor
  Serial.print(",");// adding space to organize the data
  //Serial.print("mapped value :");
  Serial.println(LEDvalue);/* displaying the scalarized value assigned to LED */
  //Serial.print("%");/* display the parentage sign */
//Serial.println("");// adding space to organize the data
 }
