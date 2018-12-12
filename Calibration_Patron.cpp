#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv/cv.h"
#include "opencv2/opencv.hpp"
#include <algorithm>
#include <sstream>
#include <string>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace cv;
using namespace std;

Mat src_gray,s_scr_gray;
Mat gray,gaussian,threshold_out,canny,contours,drawing;
Mat frame;
RNG rng(12345);
int thresh = 100;
int count_center;

float dist(Point2f a, Point2f b){
    return sqrt( pow(a.x-b.x,2.0f)+pow(a.y-b.y,2.0f) );
}

bool cmpx(Point2f a,Point2f b){
    return a.x < b.x;
}

bool cmpy(Point2f a, Point2f b){
    return a.y < b.y;
}

vector<Point2f> getControlPoints(const vector<Point2f> & centers){
    vector<Point2f> v;
    vector<int> alreadyRegistered;
    float t = 3.0f; // Valor de Desviacion Maxima
    for(int i = 0; i < centers.size();i++)
        for(int j= 0; j < centers.size(); j++){
            if(i != j && dist(centers[i],centers[j]) < t &&
            (find(alreadyRegistered.begin(), alreadyRegistered.end(),i) == alreadyRegistered.end() ||
            find(alreadyRegistered.begin(), alreadyRegistered.end(),j) == alreadyRegistered.end()) //&&
            //v.size() <= 20)
            )
            {
                // Aqui va el promedio de ambos
                float d_x = centers[i].x + centers[j].x;
                float d_y = centers[i].y + centers[j].y;
                v.push_back(Point2f(d_x/2.0f,d_y/2.0f));

                //Registramos los centros para no repetirlos
                alreadyRegistered.push_back(i);
                alreadyRegistered.push_back(j);
            }
        }

    return v;
}

void thresh_callback(int, void* )
{
    // Mat canny_output;
    // Canny( threshold_out, canny_output, thresh, thresh*2 );
    // Canny(threshold_out,canny_output,200,300,3);
    // imshow("Canny", canny);
    vector<vector<Point> > contours;
    vector<Vec4i> hierachy;
    findContours(threshold_out, contours, hierachy,RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    vector<RotatedRect> minRect( contours.size() );
    vector<RotatedRect> minEllipse( contours.size() );
    int count =0;
    for( size_t i = 0; i < contours.size(); i++ )
    {
        minRect[i] = minAreaRect( contours[i] );
        int X=minRect[i].size.width;
        int Y=minRect[i].size.height;
        int temp=abs(X-Y);
        float exc;
        //para excentricidad
        if(X>0 || Y>0){
          if(X>Y){
              exc=sqrt(pow(X,2)-pow(Y,2))/X;
          }else{
               exc=sqrt(pow(Y,2)-pow(X,2))/Y;
          }
        }

        if( contours[i].size() > 5  )
        {
            minEllipse[i] = fitEllipse(contours[i]);
            // printf("i %i distancia %i\n",i, contours[i].size());
            // printf("temp %i\n", temp);
            // printf("excentricidad %f\n", exc);
            count++;
        }
    }

     drawing = Mat::zeros( threshold_out.size(), CV_8UC3 );

    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( 200, 0, 254 );
        // contour
        drawContours( drawing, contours, (int)i, color );
        // ellipse
        //ellipse( drawing, minEllipse[i], color, 1 );
        circle(drawing,minEllipse[i].center,2,Scalar(0,200,0));
        // rotated rectangle
        Point2f rect_points[4];
        minRect[i].points( rect_points );

        // for ( int j = 0; j < 4; j++ )
        // {
        //     line( drawing, rect_points[j], rect_points[(j+1)%4], color );
        // }
    }
    //printf("cont %i\n", count);


    //Filtrar las ellipses
    cvtColor(threshold_out,src_gray, CV_GRAY2BGR);
    vector<RotatedRect> selected;
    for( int i = 0; i< contours.size(); i++ ){
        float w = minEllipse[i].size.width;
        float h = minEllipse[i].size.height;
        // float c_x = minEllipse[i].center.x;
        // float c_y = minEllipse[i].center.y;
        float dif = w - h;

        if(abs(dif) < 24){ //cantidad de elipses para filtrar
            if(hierachy[i][2] != -1){ // Si el Contour tiene Hijo que hijo sea unico
                int child_index = hierachy[i][2];
                if(hierachy[child_index][0] == -1 && hierachy[child_index][1] == -1 && hierachy[child_index][2] == -1){
                    selected.push_back(minEllipse[i]);
                    selected.push_back(minEllipse[child_index]);
                    ellipse( src_gray, minEllipse[i], Scalar(0,250,0), 1, 8 );
                    ellipse( src_gray, minEllipse[child_index], Scalar(0,255,0), 1, 8 );
                }
            }
        }
    }
    //Como minimo debemos capturar 40 elipses para continuar

    //if(selected.size() < 40) return false;
    // count=0;
    // cvtColor(threshold_out,s_scr_gray, CV_GRAY2BGR);
    // for( size_t i = 0; i< contours.size(); i++ )
    // {
    //   float w = minEllipse[i].size.width;
    //   float h = minEllipse[i].size.height;
    //   float dif = w - h;
    //     if(abs(dif) < 32){ //-->>> CAMBIAR ESTE PARAMETRO PARA FILTRAR LAS ELIPSES
    //         if(hierachy[i][2] != -1){ // Si el Contour tiene Hijo que hijo sea unico
    //             int child_index = hierachy[i][2];
    //             if(hierachy[child_index][0] == -1 && hierachy[child_index][1] == -1 && hierachy[child_index][2] == -1){
    //                 selected.push_back(minEllipse[i]);
    //                 selected.push_back(minEllipse[child_index]);
    //                 circle(s_scr_gray,minEllipse[i].center,2,Scalar(200,0,0));
    //                 putText(s_scr_gray, to_string(count), minEllipse[i].center, FONT_HERSHEY_DUPLEX, 0.5, Scalar(0,0,250), 2);
    //                 count++;
    //             }
    //         }
    //     }
    //     // for ( int j = 0; j < 4; j++ )
    //     // {
    //     //     line( drawing, rect_points[j], rect_points[(j+1)%4], color );
    //     // }
    // }

    //imshow( "numeros", s_scr_gray);

    //Extraemos los centros de todas las elipses Seleccionadas
    //cout << "Number Selected Ellipsises: " << selected.size() << endl;
    vector<Point2f> centers;
    for( int i = 0; i < selected.size(); i++ ){
        centers.push_back(selected[i].center);
    }

    vector<Point2f> CPs;
    CPs = getControlPoints(centers);
    count_center=0;
    for(int i = 0; i < CPs.size();i++){
        circle(frame,CPs[i],1,Scalar(0,0,255),3,8);
        putText(frame, to_string(count_center),CPs[i], FONT_HERSHEY_DUPLEX, 0.5, Scalar(0,250,250), 2);
        count_center++;
    }
    imshow("centros2",frame);
    //if(CPs.size() < 20) return false;
}

void preprocesamiento(Mat frame){
  //convierte a escala de grises
  cvtColor( frame,gray , COLOR_BGR2GRAY );
  //imshow("GrayScale",gray);
  //Suavizado con Gaussian BLiur
  //GaussianBlur(gray,gaussian,Size(9,9),2,2);
  GaussianBlur(gray,gaussian,Size(3,3),0);
  // Set maxValue, blockSize and c (constant value)
}

int main(int argc, char** argv )
{
      //para dividir el tempo a milisegundos
      const float CLK_TCK = 1000.0;
      //cantidad de patrones a calcular
      const int cant_patron=12;
      //cantidad de frames
      float acu_error=0;float prom_error;
      float acu_time=0;float prom_time;
      int cant_frame=0;float time;float error;
      VideoCapture cap("padron1.avi");

      // cap is the object of class video capture that tries to capture Bumpy.mp4
      if ( !cap.isOpened() )  // isOpened() returns true if capturing has been initialized.
      {
          cout << "Cannot open the video file. r \n";
          return -1;
      }
      double fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video
      // The function get is used to derive a property from the element.
      // Example:
      // CV_CAP_PROP_POS_MSEC :  Current Video capture timestamp.
      // CV_CAP_PROP_POS_FRAMES : Index of the next frame.

      while(1){

        // Mat object is a basic image container. frame is an object of Mat.
        if (!cap.read(frame)) // if not success, break loop
        // read() decodes and captures the next frame.
        {
            cout<<"\n Cannot read the video file. \n";
            break;
        }
        //Inicializamos el pipeline
         clock_t start, end;
         start = clock();
         preprocesamiento(frame);
         //int blockSize = 11;
         //double c = 3;
         double maxValue = 255;
         int blockSize = 41;
         double c = 6;
          // Adaptive Threshold
         adaptiveThreshold(gaussian, threshold_out, maxValue, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, blockSize, c);
         imshow("adaptive Threshold", threshold_out);
         // const char* source_window = "Source";
         // namedWindow( source_window );
         // imshow( source_window, threshold_out );
         // const int max_thresh = 255;
         // createTrackbar( "Canny thresh:", source_window, &thresh, max_thresh, thresh_callback );
         thresh_callback( 0, 0 );
         end = clock();
         time=(end - start) / CLK_TCK;
         acu_time=acu_time+time;

         cant_frame++;
         error=abs(count_center-cant_patron)*100/cant_patron;
         acu_error=acu_error+error;
         // printf("error %f\n", error);
         // printf("The time was: %f\n", time );
         // printf("element %i\n", count_center);
         printf("%f\n", acu_error/cant_frame);
         printf("%f\n", acu_time/cant_frame);

         imshow("Gaussian Blur", gaussian);
         imshow( "Contours", drawing );
         imshow("hierachy",src_gray);
         imshow("centros",src_gray);

         if(waitKey(30) == 27) // Wait for 'esc' key press to exit
         {
              break;
         }
      }



      return 0;
}
