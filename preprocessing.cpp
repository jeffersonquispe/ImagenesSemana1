#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv/cv.h"
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <sstream>
#include <string>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <stdarg.h>

using namespace cv;
using namespace std;

Mat src_gray,src_gray2;
Mat gray,gaussian,threshold_out,canny,contours,drawing;
Mat img_voronoi;Mat img_copy;
Mat frame;
RNG rng(12345);
int thresh = 100;
int count_center;
clock_t start, fin;
int y,x;
int cant_frame=0;
float error_count=0;
//int count=0,medX=0,medY=0,minX=1000,maxX=0,minY=1000,maxY=0;
int dist_minX,dist_minY;
//cantidad de patrones a calcular
vector<Point2f> actualPoints;
vector<Point2f> beforePoints;
float squareSizes[] = {0.02315,0.02315,0.02315,0.04540};
Size imgPixelSize = Size(640,480); // Tamaño de la imagen
vector<Point2f>points;
//para dividir el tempo a milisegundos
const float CLK_TCK = 1000.0;

void preprocesamiento(Mat frame){
  //convierte a escala de grises
  cvtColor( frame,gray , COLOR_BGR2GRAY );
  //imshow("GrayScale",gray);
  //Suavizado con Gaussian BLiur
  //GaussianBlur(gray,gaussian,Size(9,9),2,2);
  GaussianBlur(gray,gaussian,Size(3,3),0);
  // Set maxValue, blockSize and c (constant value)
}

void ShowManyImages(string title, int nArgs, ...) {
  int size;
  int i;
  int m, n;
  int x, y;

  // w - Maximum number of images in a row
  // h - Maximum number of images in a column
  int w, h;

  // scale - How much we have to resize the image
  float scale;
  int max;

  // If the number of arguments is lesser than 0 or greater than 12
  // return without displaying
  if(nArgs <= 0) {
      printf("Number of arguments too small....\n");
      return;
  }
  else if(nArgs > 14) {
      printf("Number of arguments too large, can only handle maximally 12 images at a time ...\n");
      return;
  }
  // Determine the size of the image,
  // and the number of rows/cols
  // from number of arguments
  else if (nArgs == 1) {
      w = h = 1;
      size = 300;
  }
  else if (nArgs == 2) {
      w = 2; h = 1;
      size = 300;
  }
  else if (nArgs == 3 || nArgs == 4) {
      w = 2; h = 2;
      size = 300;
  }
  else if (nArgs == 5 || nArgs == 6) {
      w = 3; h = 2;
      size = 200;
  }
  else if (nArgs == 7 || nArgs == 8) {
      w = 4; h = 2;
      size = 200;
  }
  else {
      w = 4; h = 3;
      size = 150;
  }

  // Create a new 3 channel image
  Mat DispImage = Mat::zeros(Size(100 + size*w, 60 + size*h), CV_8UC3);

  // Used to get the arguments passed
  va_list args;
  va_start(args, nArgs);

  // Loop for nArgs number of arguments
  for (i = 0, m = 20, n = 20; i < nArgs; i++, m += (20 + size)) {
      // Get the Pointer to the IplImage
      Mat img = va_arg(args, Mat);

      // Check whether it is NULL or not
      // If it is NULL, release the image, and return
      if(img.empty()) {
          printf("Invalid arguments");
          return;
      }

      // Find the width and height of the image
      x = img.cols;
      y = img.rows;

      // Find whether height or width is greater in order to resize the image
      max = (x > y)? x: y;

      // Find the scaling factor to resize the image
      scale = (float) ( (float) max / size );

      // Used to Align the images
      if( i % w == 0 && m!= 20) {
          m = 20;
          n+= 20 + size;
      }
      // Set the image ROI to display the current image
      // Resize the input image and copy the it to the Single Big Image
      Rect ROI(m, n, (int)( x/scale ), (int)( y/scale ));
      Mat temp; resize(img,temp, Size(ROI.width, ROI.height));
      temp.copyTo(DispImage(ROI));
  }

  // Create a new window, and show the Single Big Image
  namedWindow( title, 1 );
  imshow( title, DispImage);

  // End the number of arguments
  va_end(args);
}

float dist(Point2f a, Point2f b){
    return sqrt( pow(a.x-b.x,2.0f)+pow(a.y-b.y,2.0f) );
}

float StandarDesviation(const vector<float> & values ){
	int n = values.size();
    float dmean = 0.0;
    float dstddev = 0.0;

    // Mean standard algorithm
    for (int i = 0; i < n; ++i)
    {
       dmean += values[i];
    }
    dmean /= (float)n;

    // Standard deviation standard algorithm
    vector<float> var(n);
    for (int i = 0; i < n; ++i){
        var[i] = (dmean - values[i]) * (dmean - values[i]);
    }
    for (int i = 0; i < n; ++i){
        dstddev += var[i];
    }
    dstddev = sqrt(dstddev / (float)n);
    return dstddev;
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
const int cant_patron=20;

bool cmpx(Point2f a,Point2f b){
    return a.x < b.x;
}

bool cmpy(Point2f a, Point2f b){
    return a.y < b.y;
}
static void draw_point( Mat& img, Point2f fp, Scalar color )
{
    circle( img, fp, 2, color, CV_FILLED, CV_AA, 0 );
}
// Draw delaunay triangles
struct lessPoint2f
{
    bool operator()(const Point2f& lhs, const Point2f& rhs) const
    {
        return (lhs.x == rhs.x) ? (lhs.y < rhs.y) : (lhs.x < rhs.x);
    }
};

Mat delaunay(const Mat1f& points, int imRows, int imCols)
/// Return the Delaunay triangulation, under the form of an adjacency matrix
/// points is a Nx2 mat containing the coordinates (x, y) of the points
{
    map<Point2f, int, lessPoint2f> mappts;
    Mat1b adj(points.rows, points.rows, uchar(0));
    /// Create subdiv and insert the points to it
    Subdiv2D subdiv(Rect(0, 0, imCols, imRows));
    for (int p = 0; p < points.rows; p++)
    {
        float xp = points(p, 0);
        float yp = points(p, 1);
        Point2f fp(xp, yp);

        // Don't add duplicates
        if (mappts.count(fp) == 0)
        {
            // Save point and index
            mappts[fp] = p;

            subdiv.insert(fp);
        }
    }

    /// Get the number of edges
    vector<Vec4f> edgeList;
    subdiv.getEdgeList(edgeList);
    int nE = edgeList.size();
    /// Check adjacency
    for (int i = 0; i < nE; i++)
    {
        Vec4f e = edgeList[i];
        cout<<edgeList[i]<<endl;
        //cin.get();
        Point2f pt0(e[0], e[1]);
        Point2f pt1(e[2], e[3]);
        if (mappts.count(pt0) == 0 || mappts.count(pt1) == 0) {
            // Not a valid point
            continue;
        }

        int idx0 = mappts[pt0];
        int idx1 = mappts[pt1];

        // Symmetric matrix
        adj(idx0, idx1) = 1;
        adj(idx1, idx0) = 1;
    }
    return adj;
}
static void draw_delaunay( Mat& img, Subdiv2D& subdiv, Scalar delaunay_color )
{
    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    vector<Vec4f> edgeList;
    subdiv.getEdgeList(edgeList);
    Vec4f e;
    vector<Point> pt(3);
    Size size = img.size();
    Rect rect(0,0, size.width, size.height);
    for (int i = 0; i < edgeList.size(); i++)
    {
        e = edgeList[i];
        // cout<<i<<" i: "<<edgeList[i]<<endl;
        // cin.get();
    }
    // cout<<" i: "<<e<<endl;
    // cin.get();
    for( size_t i = 0; i < triangleList.size(); i++ )
    {
        Vec6f t = triangleList[i];
        // cout<<i<<" i: "<<triangleList[i]<<endl;
        // cin.get();
        pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = Point(cvRound(t[4]), cvRound(t[5]));

        // Draw rectangles completely inside the image.
        if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
        {
            line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
            line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
            line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
        }
    }
}

//Draw voronoi diagram
static void draw_voronoi( Mat& img, Subdiv2D& subdiv )
{
    vector<vector<Point2f> > facets;
    vector<Point2f> centers;
    subdiv.getVoronoiFacetList(vector<int>(), facets, centers);

    vector<Point> ifacet;
    vector<vector<Point> > ifacets(1);

    for( size_t i = 0; i < facets.size(); i++ )
    {
        ifacet.resize(facets[i].size());
        for( size_t j = 0; j < facets[i].size(); j++ )
            ifacet[j] = facets[i][j];

        Scalar color;
        color[0] = rand() & 255;
        color[1] = rand() & 255;
        color[2] = rand() & 255;
        fillConvexPoly(img, ifacet, color, 8, 0);

        ifacets[0] = ifacet;
        polylines(img, ifacets, true, Scalar(), 1, CV_AA, 0);
        circle(img, centers[i], 3, Scalar(), CV_FILLED, CV_AA, 0);
    }
}

void combinationUtil(std::vector< std::vector<int> >& v, int arr[], std::vector<int> &data, int start, int end,
                     int index, int r)
{
    if (index == r)
    {
        v.push_back(data);
        return;
    }
    for (int i=start; i<=end && end-i+1 >= r-index; i++)
    {
        data[index] = arr[i];
        combinationUtil(v, arr, data, i+1, end, index+1, r);
    }
}

void printCombination(vector< vector<int> >& v, int arr[], int n, int r)
{
    vector<int> data(r);
    combinationUtil(v, arr, data, 0, n-1, 0, r);
}

vector<vector<int> > GenerateCombinations(int n, int r){
    vector< vector<int> > v;
    int arr[n];
    for(int i = 0; i < n; i++)
        arr[i] = i;
    printCombination(v, arr, n, r);
    return v;
}

bool FindRingPattern(vector<Point2f> &probableCPs,int num_rows,int num_cols){
    int n = probableCPs.size();
    vector<Vec4f> lines;
    // Generamos todas las Combinaciones de "Lineas" en grupos de 5 posibles
    // Es 5 xq seleccionamos las lineas
    vector<vector<int> > combinations = GenerateCombinations(probableCPs.size(),num_cols);

    //Aprovechamos las lineas temporales y Selecionamos las que tengan Baja Desviacion
    vector<Vec4f> preSelectedLines;
    vector<vector<int> > combination_preSelectedLines;
    for(int i = 0; i < combinations.size();i++){
        vector<Point2f> tmpPoints(num_cols);
        Vec4f tmpLine;
        for(int j = 0; j < num_cols; j++){
            tmpPoints[j] = probableCPs[ combinations[i][j] ];
        }
        fitLine(tmpPoints,tmpLine,CV_DIST_L2,0,0.01,0.01);
        // Extraction of Features
        //Le damos forma a los valores vectoriales que nos devuelve fitline
        // r = a + r*b --> p0 punto de paso, v vector director normalizado
        float vx = tmpLine[0],vy = tmpLine[1], x0 = tmpLine[2],y0 = tmpLine[3];
        Point2f a = Point2f(x0,y0), b = Point2f(vx,vy);

        float m = 80.0;

        vector<float> distances;
        for(int k = 0; k < num_cols; k++){
            //Calculamos la distancia del punto a la recta y almacenamos para el calculo de la desviacion
            float t = ( tmpPoints[k].dot(b) - a.dot(b) ) / (norm(b) * norm(b));
            float dist = norm(tmpPoints[k] - (a + t * b));
            distances.push_back(dist);
        }

        float stddev = StandarDesviation(distances);

        //Si el error de la linea no es mucho. Seleccionamos la linea
        if(stddev < 0.5f){
            preSelectedLines.push_back(tmpLine);
            //Guardamos la Combinacion
            combination_preSelectedLines.push_back(combinations[i]);
        }

    }

    // Apply some filters here to verify line selection
    // Then Order Points and Store in CPs(Hard verification of only 20 Ordered and Aligned Control Points)
    // Acordemonos que ya seleccionamos solo lineas con 5 puntos
    if(preSelectedLines.size() == 4){
        //Tenemos que ordenar las lineas. (Recordemos que son lineas paralelas)
        //Primero verificamos la pendiente

        //LINE ORDERING
            //Recordemos la grilla que presenta openCV
            // -------> x+
            // |
            // |
            // y+

            Vec4f Line = preSelectedLines[0];
            float vx = Line[0],vy = Line[1], x0 = Line[2],y0 = Line[3];
            //Pendiente
            float slope = vy/vx;
            if(abs(slope) < 5.0f){ //Evaluamos las pendientes de casi 80 grados (Revisar esta funcion)
                std::vector<float> y_intersection(4);
                //Calcular el punto de interseccion por el eje y
                for(int i = 0; i < 4; i++){
                    Vec4f tmpLine = preSelectedLines[0];
                    float vx = tmpLine[0],vy = tmpLine[1], x0 = tmpLine[2],y0 = tmpLine[3];

                    float t = -x0 / vx;
                    float y = y0 + t*vy;

                    y_intersection[i] = y;
                }
                //Realizamos un bubble sort en base a las intersecciones con el eje y
                //ordenamiento por burbuja
                bool swapp = true;
                while(swapp)
                {
                    swapp = false;
                    for (int i = 0; i < preSelectedLines.size()-1; i++)
                    {
                        if (y_intersection[i] > y_intersection[i+1] ){
                            //Cambiamos en todos nuestros vectores
                            std::swap(y_intersection[i],y_intersection[i+1]);
                            std::swap(preSelectedLines[i],preSelectedLines[i+1]);
                            std::swap(combination_preSelectedLines[i],combination_preSelectedLines[i+1]);
                            swapp = true;
                        }
                    }
                }// Fin del ordenamiento
                // Para Cada Linea obtener los CP segun la combinacion y ordenarlos por el eje X
                // Obtenemos los puntos desde el CP
                std::vector<Point2f> tmpCPs;
                for(int i = 0; i < num_rows; i++){
                    std::vector<Point2f> tmpCenters(num_cols);
                    for(int j = 0; j < num_cols; j++){
                        tmpCenters[j] = probableCPs[ combination_preSelectedLines[i][j] ];
                    }
                    sort(tmpCenters.begin(), tmpCenters.end(),cmpx);
                    for(int j = 0; j < num_cols; j++){
                        tmpCPs.push_back(tmpCenters[j]);
                    }
                }
                cout << tmpCPs << '\n';
                //cin.get();
                probableCPs.clear();
                probableCPs = tmpCPs;

                return true;
            }
    }
    return false;
}// fin de funcion importante
void calcBoardCornerPositions(cv::Size size, float squareSize, std::vector<cv::Point3f> &corners, int patternType){
    corners.clear();
    switch(patternType){
        case CHESSBOARD:
        case CIRCLES_GRID:
        case RINGS_GRID:
            for(int i = 0 ; i < size.height; i++)
                for(int j = 0; j < size.width; j++)
                    corners.push_back(cv::Point3f( float(j * squareSize),float(i*squareSize),0) );
            break;
        case ASYMMETRIC_CIRCLES_GRID:
            for( int i = 0; i < size.height; i++ )
                for( int j = 0; j < size.width; j++ )
                    corners.push_back(Point3f(float((2*j + i % 2)*squareSize), float(i*squareSize), 0));
            break;
    }
}
double computeReprojectionErrors(const std::vector< std::vector<cv::Point3f> >& objectPoints,
                                    const std::vector< std::vector<cv::Point2f> >& imagePoints,
                                    const std::vector<cv::Mat>& rvecs,const std::vector<cv::Mat>& tvecs,
                                    const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
                                    std::vector<float> & perFrameErrors){

    std::vector<cv::Point2f> imagePoints2;
    size_t totalPoints = 0;
    double totalErr = 0, err;
    perFrameErrors.resize(objectPoints.size());

    for(size_t i = 0; i < objectPoints.size(); ++i ){

        cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);

        err = norm(imagePoints[i], imagePoints2, NORM_L2);

        size_t n = objectPoints[i].size();
        perFrameErrors[i] = (float) std::sqrt(err*err/n);
        totalErr        += err*err;
        totalPoints     += n;
    }

    return std::sqrt(totalErr/totalPoints);

}
bool thresh_callback(int, void* ,std::vector<cv::Point2f>& points1,bool& isTracking,std::vector<cv::Point2f>& oldPoints)
{
    // Mat canny_output;
    // Canny( threshold_out, canny_output, thresh, thresh*2 );
    // Canny(threshold_out,canny_output,200,300,3);
    // imshow("Canny", canny);
    int medX=0,medY=0,minX=1000,maxX=0,minY=1000,maxY=0;
    vector<vector<Point> > contours;
    vector<Vec4i> hierachy;
    //encuentra los contornos
    findContours(threshold_out, contours, hierachy,RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    vector<RotatedRect> minRect( contours.size() );
    vector<RotatedRect> minEllipse( contours.size() );
    int count =0;

    for( size_t i = 0; i < contours.size(); i++ )
    {
        minRect[i] = minAreaRect( contours[i] );
        //incialmente se queria adjuntar excentricidad pero solo es util para
        //hough circlespara excentricidad
        // int X=minRect[i].size.width;
        // int Y=minRect[i].size.height;
        // int temp=abs(X-Y);
        // float exc;
        // if(X>0 || Y>0){
        //   if(X>Y){
        //       exc=sqrt(pow(X,2)-pow(Y,2))/X;
        //   }else{
        //        exc=sqrt(pow(Y,2)-pow(X,2))/Y;
        //   }
        // }
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
    Scalar color = Scalar( 200, 0, 254 );
    // for( size_t i = 0; i< contours.size(); i++ )
    // {
    //     // contour
    //     drawContours( drawing, contours, (int)i, color );
    //     // ellipse
    //     //ellipse( drawing, minEllipse[i], color, 1 );
    //     circle(drawing,minEllipse[i].center,2,Scalar(0,200,0));
    //     // rotated rectangle
    //     Point2f rect_points[4];
    //     minRect[i].points( rect_points );
    //
    //     // for ( int j = 0; j < 4; j++ )
    //     // {
    //     //     line( drawing, rect_points[j], rect_points[(j+1)%4], color );
    //     // }
    // }
    //printf("cont %i\n", count);

    //Filtrar las ellipses
    cvtColor(threshold_out,src_gray, CV_GRAY2BGR);
    cvtColor(threshold_out,src_gray2, CV_GRAY2BGR);
    vector<RotatedRect> selected;
    for( int i = 0; i< contours.size(); i++ ){
        float width = minEllipse[i].size.width;
        float height = minEllipse[i].size.height;
        float dif = width - height;
        //printf("- %f - %f\n",w,h );
        //cin.get( );
        if(abs(dif) < cant_patron*2){ //cantidad de elipses para filtrar
            if(hierachy[i][2] != -1){ // Si el Contour tiene Hijo que hijo sea unico
                //cout << hierachy[i] << '\n';
                //cin.get( );
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

    //Extraemos los centros de todas las elipses Seleccionadas
    //cout << "Number Selected Ellipsises: " << selected.size() << endl;
    vector<Point2f> centers;

    for( int i = 0; i < selected.size(); i++ ){
      if(selected[i].center.x==0 && selected[i].center.y==0){
          continue;
      }else{
        centers.push_back(selected[i].center);
      }
    }

    vector<Point2f> CenterPoints;
    vector<Point2f> CenterPointsFilter;
    CenterPoints = getControlPoints(centers);
    count_center=0;
    //cin.get();

    for(int i = 0; i < CenterPoints.size();i++){
      y=CenterPoints[i].y;
      x=CenterPoints[i].x;
      count_center++;
      minX=min(minX,x);
      minY=min(minY,y);
      maxX=max(maxX,x);
      maxY=max(maxY,y);
      medX=medX+x;
      medY=medY+y;
    }

    medX=medX/count_center;
    medY=medY/count_center;
    dist_minX=min(abs(medX-minX),abs(medX-maxX));
    dist_minY=min(abs(medY-minY),abs(medY-maxY));
    minX=medX-dist_minX-25;minY=medY-dist_minY-25;
    maxX=medX+dist_minX+25;maxY=medY+dist_minY+25;
    //printf(" distancia Xmin %d distancia Ymin %d   %d",dist_minX,dist_minY,count_center);
    circle(src_gray2,Point(medX,medY),8,Scalar(180,0,200),-1);
    circle(src_gray2,Point(medX+dist_minX+15,medY+dist_minY+15),8,Scalar(180,0,0),-1);
    circle(src_gray2,Point(medX+dist_minX+15,medY-dist_minY-15),8,Scalar(180,0,0),-1);
    circle(src_gray2,Point(medX-dist_minX-15,medY+dist_minY+15),8,Scalar(180,0,0),-1);
    circle(src_gray2,Point(medX-dist_minX-15,medY-dist_minY-15),8,Scalar(180,0,0),-1);
    vector<Point2f> centers2;
    for( int i = 0; i < selected.size(); i++ ){
      if(selected[i].center.x>minX && selected[i].center.x<maxX &&
         selected[i].center.y>minY && selected[i].center.y<maxY){
           centers2.push_back(selected[i].center);
      }
    }
    CenterPointsFilter = getControlPoints(centers2);

    for(int i=0;i<CenterPointsFilter.size();i++){
          circle(src_gray2,CenterPointsFilter[i],3,Scalar(180,0,200),-1);
          //printf(" i %s\n",to_string(i));
          //line(frame,CenterPointsFilter[i],CenterPointsFilter[i+1],(0,255,0),5);
          //putText(frame, to_string(i), CenterPointsFilter[i], FONT_HERSHEY_DUPLEX, 0.5, Scalar(255,255,255), 2);
          //printf(" (y %i  x %i)- ",y,x);
        //cin.get();
    }
     fin = clock();
    Mat img_copy = frame.clone();
    //================Ordenar Puntos===================================
    // float dist_temp,min_dist=1000;
    // int pos;
    // actualPoints=CenterPointsFilter;
    // if(cant_frame==0){beforePoints=CenterPointsFilter;}
    // cout << "/* message */"<< actualPoints<< '\n';
    // cin.get();
    // for(int i=0;i<actualPoints.size();i++){
    //   pos=0;
    //   for(int j=0;j<beforePoints.size();j++){
    //     dist_temp=dist(actualPoints[i],beforePoints[j]);
    //     min_dist=min(dist_temp,min_dist);
    //     // if(j==1){
    //     // cout <<j <<" temp "<< min_dist<< '\n';
    //     // cin.get();
    //     // }
    //     // if(dist_temp!=min_dist){
    //     //   pos=j;
    //     // }
    //   }
    //   actualPoints[i]=actualPoints[pos];
    //   beforePoints[i]=actualPoints[i];
    // }
    //======================================================================
    //=============DALONE==================================
    // Define window names
    string win_delaunay = "Delaunay Triangulation";
    string win_voronoi = "Voronoi Diagram";

    // Turn on animation while drawing triangles
    bool animate = true;
    // Define colors for drawing.
    Scalar delaunay_color(255,255,255), points_color(0, 0, 255);

    // // Read in the image.
    // Mat img = imread("image.jpg");
    //
    // // Keep a copy around
    // Mat img_orig = img.clone();

    // Rectangle to be used with Subdiv2D
    Size size = frame.size();
    Rect rect(0, 0, size.width, size.height);

    // Create an instance of Subdiv2D
    Subdiv2D subdiv(rect);

    // Create a vector of points.
    vector<Point2f> points;
    points=CenterPointsFilter;

    // Insert points into subdiv
    for( vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
    {
        subdiv.insert(*it);
        // Show animation
        if (animate)
        {
            //Draw delaunay triangles
            draw_delaunay( img_copy, subdiv, delaunay_color );
          //  imshow(win_delaunay, img_copy);
            //waitKey(160);
        }
        //cout << "delano"<< subdiv<< '\n';
        //cin.get();
    }

    // Draw delaunay triangles
    //draw_delaunay(frame, subdiv, delaunay_color);
    //=========================Delanauy ============================
    // Mat1f pointsEdges=Mat1f(CenterPointsFilter.size(),2CV_64F,Crds.data());
    // Mat1b adj = delaunay(pointsEdges, size.width, size.height);
    // for (int i = 0; i < pointsEdges.rows; i++)
    // {
    //     int xi = pointsEdges.at<float>(i, 0);
    //     int yi = pointsEdges.at<float>(i, 1);
    //     /// Draw the edges
    //     for (int j = i + 1; j < pointsEdges.rows; j++)
    //     {
    //         if (adj(i, j))
    //         {
    //             int xj = pointsEdges(j, 0);
    //             int yj = pointsEdges(j, 1);
    //             line(frame, Point(xi, yi), Point(xj, yj), Scalar(255, 0, 0), 1);
    //         }
    //     }
    // }
    //
    // for (int i = 0; i < pointsEdges.rows; i++)
    // {
    //     int xi = points(i, 0);
    //     int yi = points(i, 1);
    //
    //     /// Draw the nodes
    //     circle(frame, Point(xi, yi), 1, Scalar(0, 0, 255), -1);
    // }
    //
    // imshow("im", im);
    //=============================================================
    // Draw points
    for( vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
    {
      //  draw_point(frame, *it, points_color);
    }

    // Allocate space for Voronoi Diagram
    img_voronoi = Mat::zeros(frame.rows, frame.cols, CV_8UC3);

    // Draw Voronoi diagram
    draw_voronoi( img_voronoi, subdiv );

    // Show results.
    //imshow( win_delaunay, frame);
    //imshow( win_voronoi, img_voronoi);
    //waitKey(0);
    //======================================================
    // vector<Point2f>oldPoints;

    vector<Point2f> trackedPoints;
    int numTrackedItems = 20;
    //bool isTracking=true;

    if(isTracking){
      trackedPoints.resize(numTrackedItems);
      vector<float> distances;

      for(int k = 0; k < numTrackedItems;k++){
          Point2f tmp = oldPoints[k];
          float min = 100000.0f;
          int index = 0;
          for(int i = 0; i < CenterPoints.size(); i++){
              if( min > dist(oldPoints[k],CenterPoints[i]) ){
                  min = dist(oldPoints[k],CenterPoints[i]);
                  index = i;
              }
          }
          distances.push_back(dist(oldPoints[k],CenterPoints[index]));
          trackedPoints[k] = CenterPoints[index]; // Actualizamos la posicion de los puntos
      }
      bool isCorrect = true;

      float dstddev = StandarDesviation(distances);

      //Aumentar validaciones en esta zona
      if(dstddev > 3.0f)
          isCorrect = false;

      //Revisar que np haya duplicados
      for(int i = 0; i < trackedPoints.size()-1;i++)
          for(int j = i+1; j < trackedPoints.size();j++)
              if(trackedPoints[i] == trackedPoints[j])
                  isCorrect = false;

      //Si no es correcto el mandar señal para tratar de capturar el tracking
      if(!isCorrect){
          cout << "Couldnt keep tracking\n";
          // cin.get();
          isTracking = false;
      }

  }

  // isTracking = false;
  // if(!isTracking){
  else{
      cout << "Start Tracking\n";

      // Buscamos encontrar el patron, devolvemos solo el numero correspondiente de nodos
      // Ademas Ordenamos los nodos, primero por fila, luego por columna
      bool patternWasFound = FindRingPattern(CenterPoints,4,5);
      //patternWasFound = false;
      //Esta parte del codigo debe enviar 20 puntos Ordenados y en grilla hacia TrackedPoints
      //En cualquier otro caso debe pasar al siguiente frame y tratar otra vez
      //El ordenamiento a pasar es el siguiente

      if(patternWasFound){
          trackedPoints.clear();
          for(int i = 0; i < numTrackedItems; i++){
              trackedPoints.push_back(CenterPoints[i]);
              //circle(src_gray2,CenterPoints[i],5,Scalar(255,0,0),3,8);
              putText(src_gray2, to_string(i), CenterPointsFilter[i], FONT_HERSHEY_DUPLEX, 0.5, Scalar(255,255,0), 2);
              //cin.get();
          }
          //cin.get();
          isTracking = true;
      }
  }

  imshow("ok",src_gray2);

  // Copiamos el vector a points que seran nuestros CPs
  points1 = trackedPoints;
  return isTracking;
}

int main(int argc, char** argv )
{
      vector<Mat> images;
      //cantidad de frames
      float acu_error=0;float prom_error;
      float acu_time=0;float prom_time;
      float time;
      //Read a video
      VideoCapture cap("padron2.avi");
      //read a Image about metrics
      Mat imagen=imread("presentacion.png",CV_LOAD_IMAGE_COLOR);
      vector<Point2f> PointBuffer;
      vector<Point2f> oldPoints; // Punto usados para el Tracking en RingGrid
      bool isTracking = false; // Variable que ayuda a FindRingPatterns
      Size patternSize[] = {Size(9,6),Size(4,5),Size(4,11),Size(5,4)};
      enum {CAPTURING = 0, CALIBRATED = 1 };
      // cap is the object of class video capture that tries to capture Bumpy.mp4
      if ( !cap.isOpened() )  // isOpened() returns true if capturing has been initialized.
      {
          cout << "Cannot open the video file. r \n";
          return -1;
      }

      double fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video
      // The function get is used to derive a property from the element.
      Mat cameraMatrix = cv::Mat::eye(3,3,CV_64F); // Matriz para guardar la camera Intrinsics
      Mat distCoeffs = cv::Mat::zeros(8, 1,CV_64F); // Aqui guardamos los coeficientes de Distorsion
      vector<Mat> rvecs,tvecs;
      double rms, totAvgErr;
      int noImages = 25;
      vector<float> reprojErrors;
      int sample_FPS = 40;
      vector< vector<Point3f> > objPoints;
      vector<vector<Point2f> > imgPoints;
      int key;
      int counter = 0;
      int mode = 0;
      while(1){
        // Mat object is a basic image container. frame is an object of Mat.
        if (!cap.read(frame)) // if not success, break loop
        // read() decodes and captures the next frame.
        {
            cout<<"\n Cannot read the video file. \n";
            break;
        }
        //================================Inicializamos el pipeline=================================
         preprocesamiento(frame);
         //Adaptive Threshold values
         double maxValue = 255;
         int blockSize = 41;
         double c = 6;
         //int blockSize = 11;
         //double c = 3;
         //Adaptive Threshold
         start = clock();
         adaptiveThreshold(gaussian, threshold_out, maxValue, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, blockSize, c);
         //imshow("adaptive Threshold", threshold_out);
         // const char* source_window = "Source";
         // namedWindow( source_window );
         // imshow( source_window, threshold_out );
         // const int max_thresh = 255;
         // createTrackbar( "Canny thresh:", source_window, &thresh, max_thresh, thresh_callback );
         switch(mode){
           case CAPTURING:{
               objPoints.resize(1);
               calcBoardCornerPositions(patternSize[3],squareSizes[3],objPoints[0],3);
               objPoints.resize(noImages,objPoints[0]);
               bool found=false;
               found=thresh_callback( 0, 0 , PointBuffer,isTracking,oldPoints);
               if(isTracking)
                   oldPoints = PointBuffer;
              if(found){
                //avgColinearityPreCalibrated.push_back( getAvgColinearityFromVector(PointBuffer, patternSize[patternType]));
                drawChessboardCorners(frame, patternSize[3], PointBuffer,found);
                if(counter % sample_FPS == 0){
                        imgPoints.push_back(PointBuffer);
                        cout << "Imagen Capturada. Imagenes Totales: " << imgPoints.size() << endl;
                }
              }
              if(imgPoints.size() == noImages)
        				mode++;
        			else
        				key = waitKey(1); // Tiempo de espera 1000ms
                break;
              }
           case CALIBRATED:{
             rms = calibrateCamera(objPoints, imgPoints, imgPixelSize, cameraMatrix, distCoeffs, rvecs, tvecs);
             cout << "avg_reprojection_error" << rms << endl;
             cout << "camera_matrix" << endl << cameraMatrix << endl;
             cout << "Distortion_coefficients" << endl << distCoeffs << endl;

             totAvgErr = computeReprojectionErrors(objPoints,imgPoints,rvecs, tvecs,cameraMatrix,distCoeffs, reprojErrors);
             cout << "Average Reprojection Error: " << totAvgErr << endl;

             cout << "Empezando la Distorsion \n";
                break;}
         }

         fin = clock();
         time=(fin - start) / CLK_TCK;
         acu_time=acu_time+time;

         cant_frame++;
         printf("frames %i\n",cant_frame);
         //error=abs(count_center-cant_patron)*100/cant_patron;
         acu_error=acu_error+error_count;
         // printf("error %f\n", error);
         // printf("The time was: %f\n", time );
         // printf("element %i\n", count_center);
         int rowsg=gray.rows;
         int colg=gray.cols;
         int rowsd=drawing.rows;
         int cold=drawing.cols;
         //vector<Mat>images{gray, gray, gray};
         //displayWindows(images);
         //hconcat(gray,drawing,final1);
         imshow("final",frame);

         if(cant_frame%10==0){
           imagen=imread("presentacion.png",CV_LOAD_IMAGE_COLOR);
           putText(imagen, "Error: "+to_string(acu_error/cant_frame) + " %", Point(300,300), FONT_HERSHEY_DUPLEX, 2, Scalar(0,0,250), 6);
           putText(imagen, "Time: "+to_string(acu_time/cant_frame) + " ms", Point(300,400), FONT_HERSHEY_DUPLEX, 2, Scalar(0,250,0), 6);
           putText(imagen, "Accuracy: "+to_string(100-acu_error/cant_frame) + " ms", Point(300,500), FONT_HERSHEY_DUPLEX, 2, Scalar(0,0,250), 6);
         }
         ShowManyImages("Image", 5, src_gray,src_gray2,frame,img_voronoi,imagen);
         printf("rws %i %i \n", rowsg,rowsd);
         printf(" colg %i  %i \n", colg,cold);
         printf("gray %f\n", acu_error/cant_frame);
         printf("%f\n", acu_time/cant_frame);

         /*if(waitKey(1) == 27) // Wait for 'esc' key press to exit
         {
              break;
         }*/

         waitKey(1);
      }

      return 0;
}
