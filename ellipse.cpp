#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;
Mat src_gray;
int thresh = 100;
RNG rng(12345);
void thresh_callback(int, void* );

int main( int argc, char** argv )
{
  VideoCapture cap("PadronAnillos_02.avi");
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

  namedWindow("A_good_name",CV_WINDOW_AUTOSIZE);
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

      cvtColor( src, src_gray, COLOR_BGR2GRAY );
      //blur( src_gray, src_gray, Size(5,5) );
      const char* source_window = "Source";
      namedWindow( source_window );
      imshow( source_window, src );
      const int max_thresh = 255;
      createTrackbar( "Canny thresh:", source_window, &thresh, max_thresh, thresh_callback );
      thresh_callback( 0, 0 );
      if(waitKey(30) == 27) // Wait for 'esc' key press to exit
      {
          break;
      }
    }
    return 0;

}
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
        //minRect[i] = minAreaRect( contours[i] );
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
        //drawContours( drawing, contours, (int)i, color );
        // ellipse
        ellipse( drawing, minEllipse[i], color, 1 );
        // rotated rectangle
        Point2f rect_points[4];
        //minRect[i].points( rect_points );
        // for ( int j = 0; j < 4; j++ )
        // {
        //     line( drawing, rect_points[j], rect_points[(j+1)%4], color );
        // }
    }
    imshow( "Contours", drawing );
}
