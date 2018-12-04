#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <stdio.h>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv/cv.h"
#include "opencv2/opencv.hpp"
#include <algorithm>
#include <sstream>
#include <string>
// highgui - an interface to video and image capturing.


using namespace cv;
using namespace std;
// Namespace where all the C++ OpenCV functionality resides.

using namespace std;
Mat src_gray,s_scr_gray;
int thresh = 100;
RNG rng(12345);

void thresh_callback(int, void* )
{
    Mat canny_output;
    Canny( src_gray, canny_output, thresh, thresh*2 );
    vector<vector<Point> > contours;
    findContours( canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
    vector<RotatedRect> minRect( contours.size() );
    vector<RotatedRect> minEllipse( contours.size() );
    for( size_t i = 0; i < contours.size(); i++ )
    {
        minRect[i] = minAreaRect( contours[i] );
        if( contours[i].size() > 5 )
        {
            minEllipse[i] = fitEllipse( contours[i] );
        }
    }
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        // contour
        drawContours( drawing, contours, (int)i, color );
        // ellipse
        ellipse( drawing, minEllipse[i], color, 2 );
        // rotated rectangle
        Point2f rect_points[4];
        minRect[i].points( rect_points );
        for ( int j = 0; j < 4; j++ )
        {
            line( drawing, rect_points[j], rect_points[(j+1)%4], color );
        }
    }
    imshow( "Contours", drawing );
}
Mat thresh_callback2(int, void*)
{
    Mat canny_output;
    Canny( src_gray, canny_output, thresh, thresh*2 );
    vector<vector<Point> > contours;
    findContours( canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE );
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<Point2f>centers( contours.size() );
    vector<float>radius( contours.size() );
    for( size_t i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( contours[i], contours_poly[i], 3, true );
        boundRect[i] = boundingRect( contours_poly[i] );
        minEnclosingCircle( contours_poly[i], centers[i], radius[i] );
    }
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        //drawContours( drawing, contours_poly, (int)i, color );
        //rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2 );
        circle( drawing, centers[i], (int)radius[i], color, 1 );
    }
    imshow( "Contours", drawing );
}

int main(int argc, char** argv )
{
    VideoCapture cap("PadronAnillos_02.avi");
    // cap is the object of class video capture that tries to capture Bumpy.mp4
    if ( !cap.isOpened() )  // isOpened() returns true if capturing has been initialized.
    {
        cout << "Cannot open the video file. \n";
        return -1;
    }
    double fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video
    // The function get is used to derive a property from the element.
    // Example:
    // CV_CAP_PROP_POS_MSEC :  Current Video capture timestamp.
    // CV_CAP_PROP_POS_FRAMES : Index of the next frame.

    namedWindow("A_good_name",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"
    // first argument: name of the window.
    // second argument: flag- types:
    // WINDOW_NORMAL : The user can resize the window.
    // WINDOW_AUTOSIZE : The window size is automatically adjusted to fitvthe displayed image() ), and you cannot change the window size manually.
    // WINDOW_OPENGL : The window will be created with OpenGL support.
    Mat final1,final2,final3,tc;
    while(1)
    {
        Mat frame;
        // Mat object is a basic image container. frame is an object of Mat.
        if (!cap.read(frame)) // if not success, break loop
        // read() decodes and captures the next frame.
        {
            cout<<"\n Cannot read the video file. \n";
            break;
        }

    Mat src = frame;
    Mat gray,s_gray,thresh;
//
// /*=======================Elipses=========================================*/
         // cvtColor(src, gray, COLOR_BGR2GRAY);
         // //medianBlur(gray, gray, 5);
         // //GaussianBlur(gray,s_gray,Size(9,9),0);
         // //imshow("GaussianBlur", s_gray );
         // vector<Vec3f> circles;
         // HoughCircles(s_gray, circles, HOUGH_GRADIENT, 0.5,
         //              s_gray.rows/16,  // change this value to detect circles with different distances to each other
         //              100, 30, 1, 50 // change the last two parameters
         //         // (min_radius & max_radius) to detect larger circles
         // );
         // int count=0;
         // for( size_t i = 0; i < circles.size(); i++ )
         // {
         //     Vec3i c = circles[i];
         //     Point center = Point(c[0],c[1]);
         //     // circle center
         //     circle( src, center, 1, Scalar(0,10,20), 1, LINE_AA);
         //     // circle outline
         //     int radius = c[2];
         //     printf("x %d y %d i %d\n ",c[0],c[1],i);
         //     circle( src, center, radius, Scalar(254,254,0), 1, LINE_AA);
         //     count++;
         //     //system.pause;
         // }
         // printf("cantidad circulos %d\n ",count);
         // imshow("oki", frame);
         // cin.get();
// /*=======================CIRCULOS=========================================*/
         // cvtColor(src,gray, COLOR_BGR2GRAY);
         // //medianBlur(gray, gray, 5);
         // GaussianBlur(gray,s_gray,Size(21,21),0);
         // //imshow("GaussianBlur", s_gray );
         // vector<Vec3f> circles;
         // HoughCircles(s_gray, circles, HOUGH_GRADIENT, 0.5,
         //              s_gray.rows/16,  // change this value to detect circles with different distances to each other
         //              100, 30, 1, 50 // change the last two parameters
         //         // (min_radius & max_radius) to detect larger circles
         // );
         // int count=0;
         // for( size_t i = 0; i < circles.size(); i++ )
         // {
         //     Vec3i c = circles[i];
         //     Point center = Point(c[0],c[1]);
         //     // circle center
         //     circle( s_gray, center, 1, Scalar(0,10,20), 1, LINE_AA);
         //     // circle outline
         //     int radius = c[2];
         //     printf("x %d y %d i %d\n ",c[0],c[1],i);
         //     circle( s_gray, center, radius, Scalar(254,254,0), 1, LINE_AA);
         //     count++;
         //     //system.pause;
         // }
         // printf("cantidad circulos %d\n ",count);
         // imshow("oki", s_gray);
         // cin.get();
// /* =======================CONTORNOS=====================*/
         // if( src.empty() )
         // {
         //     cout << "Could not open or find the image!\n" << endl;
         //     cout << "Usage: " << argv[0] << " <Input image>" << endl;
         //     return -1;
         // }
         //  cvtColor( src, src_gray, COLOR_BGR2GRAY );
         //  blur( src_gray, src_gray, Size(3,3) );
         //  const char* source_window = "Source";
         //  namedWindow( source_window );
         //  imshow( source_window, src );
         //  const int max_thresh = 255;
         //  createTrackbar( "Canny thresh:", source_window, &thresh, max_thresh, thresh_callback );
         //  thresh_callback( 0, 0 );

         //hconcat(s_gray, s_scr_gray,final1);
         //hconcat(frame, tc,final2);         //vconcat(final1, final2,final3);
         //imshow("A_good_name", c);
        // imshow("2", final2);
/* ======================canny=================================*/
        //asign frame variables
         Mat gaussian_gray,blur_gray,median_gray,bilat_gray,final1,final2,final3;
         int y,x;
         int count=0,medX=0,medY=0,minX=1000,maxX=0,minY=1000,maxY=0;
         int dist_minX,dist_minY;
         //create objet keypoint to points
         vector<KeyPoint> keypoints;
         //Converts an image from one color space to another.
         cvtColor( frame, src, COLOR_BGR2GRAY );
         //Gaussian Blur filter
         //GaussianBlur(src,gaussian_gray,Size(3,3),0);
         //imshow("Gaussian Blur", gaussian_gray);
         // Gaussian Blur filter
          blur(src,blur_gray,Size(9,9),Point(-1,-1));
         //imshow("Image Blur", blur_gray);
         //Gaussian Blur filter
         medianBlur(src,median_gray,9);
         //imshow("median Blur", median_gray);
         //Gaussian Blur filter
         bilateralFilter(src,gaussian_gray,3,3*2,3/2);
         //imshow("Bilateral Blur", bilat_gray);
         //dibuja los iamgenes de filtros
         // hconcat(gaussian_gray, blur_gray,final1);
         // //imshow("gaussian-blur", final1);
         // hconcat(median_gray, bilat_gray,final2);
         // //imshow("median-bilat", final2);
         // vconcat(final1,final2,final3);
         // imshow("gaussian-blur-median-bilat", final3);

         Canny(gaussian_gray,gaussian_gray,200,300,3);
         SimpleBlobDetector::Params params;
         params.minThreshold=10;
         params.maxThreshold=200;
         Ptr<SimpleBlobDetector> detector=SimpleBlobDetector::create(params);
         detector->detect(gaussian_gray,keypoints);
         //Mat drawI=gaussian_gray.clone();
         //drawKeypoints( src, keypoints, s_gray );
          for(size_t i=0;i<keypoints.size();i++){
           //CACULATE points to maximun, minimun an center points near rings
           y=keypoints[i].pt.y;
           x=keypoints[i].pt.x;
           count++;
           minX=min(minX,x);minY=min(minY,y);
           maxX=max(maxX,x);maxY=max(maxY,y);
           medX=medX+x;
           medY=medY+y;
         }
         medX=medX/count;
         medY=medY/count;
         //hallar las distancias minimas
         dist_minX=min(abs(medX-minX),abs(medX-maxX));
         dist_minY=min(abs(medY-minY),abs(medY-maxY));
         printf(" distancia Xmin %d distancia Ymin %d",dist_minX,dist_minY);
         // circle(frame,Point(medX,medY),3,Scalar(180,0,200),-1);
         // circle(frame,Point(medX+dist_minX+20,medY+dist_minY+20),3,Scalar(180,0,200),-1);
         // circle(frame,Point(medX+dist_minX+20,medY-dist_minY-20),3,Scalar(180,0,200),-1);
         // circle(frame,Point(medX-dist_minX-20,medY+dist_minY+20),3,Scalar(180,0,200),-1);
         // circle(frame,Point(medX-dist_minX-20,medY-dist_minY-20),3,Scalar(180,0,200),-1);
         minX=medX-dist_minX-20;minY=medY-dist_minY-20;
         maxX=medX+dist_minX+20;maxY=medY+dist_minY+20;
         //vaciar el contar para enuamrar los puntos verdaderos
         count=0;
         for(int i=0;i<keypoints.size();i++){
           y=keypoints[i].pt.y;
           x=keypoints[i].pt.x;
           if(x>minX && x<maxX && y>minY && y<maxY){
             circle(gaussian_gray,keypoints[i].pt,3,Scalar(180,0,200),-1);
             circle(frame,keypoints[i].pt,3,Scalar(180,0,200),-1);
             printf(" i %s\n",to_string(i));
             putText(frame, to_string(i), Point(x,y), FONT_HERSHEY_DUPLEX, 0.5, Scalar(255,255,255), 2);
             printf(" (y %d  x %d)- ",y,x);
             count++;
           }
         }
         // printf("(width %d, Height %d)\n",gaussian_gray.cols,gaussian_gray.rows );
         imshow("GaussianBlur", gaussian_gray);
         imshow("Canny2", frame);
         printf("total %d\n", count);
         //cin.get();
        //first argument: name of the window.
        //second argument: image to be shown(Mat object).
        if(waitKey(30) == 27) // Wait for 'esc' key press to exit
        {
            break;
        }
    }

    return 0;
}
// END OF PROGRAM
