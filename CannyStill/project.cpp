
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cuda.hpp>

using namespace std;
using namespace cv;

int thresh = 50;
IplImage* img = NULL;
IplImage* img0 = NULL;
IplImage* img2 = NULL;
IplImage* img2_0 = NULL;
IplImage* img3 = NULL;
IplImage* img3_0 = NULL;
IplImage* img4 = NULL;
IplImage* img4_0 = NULL;
//storage0 is used to store the number of contours
CvMemStorage* storage0 = NULL;
//storage 1 used to store square
CvMemStorage* storage = NULL;
//store rectangles
CvMemStorage* storage2 = NULL;
//store circles
CvMemStorage* storage3 = NULL;
//store diamonds
CvMemStorage* storage4 = NULL;
//store polygons
CvMemStorage* storage5 = NULL;
//store triangles
CvMemStorage* storage6 = NULL;
//this three dimension arry is used for storing the information of the polygons, which in order is: the number of the image, the polys of the polygon, and the convexity of the contours.
static int polygon[4][10][2];

const char * wndname = "final project";
//this function is used for detect the angle by using cosine theorem, return the value of the cosine.
double angle(CvPoint* pt1, CvPoint* pt2, CvPoint* pt0)
{
	double dx1 = pt1->x - pt0->x;
	double dy1 = pt1->y - pt0->y;
	double dx2 = pt2->x - pt0->x;
	double dy2 = pt2->y - pt0->y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

//this function is used for detect the length between three points and detect whether they are the same. if so, return 1, else reutrn 0;
double distance(CvPoint* pt1, CvPoint* pt2, CvPoint* pt0)
{
	double dx1 = pt1->x - pt0->x;
	double dy1 = pt1->y - pt0->y;
	double dx2 = pt2->x - pt0->x;
	double dy2 = pt2->y - pt0->y;
	double distance1 = sqrt((dx1*dx1 + dy1*dy1));
	double distance2 = sqrt((dx2*dx2 + dy2*dy2));
	if (((distance1 - 10) <= distance2)&&
		(distance2 <= (distance1 + 10)))
	{
		return 1;
	}
	return 0;

}
//this function is used for finding all the contours in the picture
//the return value should be the number of contours
int count_the_contours(IplImage* img, CvMemStorage* storage)
{
	CvSeq* contours;
	int i, c, cnt;
	CvSize sz = cvSize(img->width & -2, img->height & -2);

	IplImage* timg = cvCloneImage(img);
	IplImage* gray = cvCreateImage(sz, 8, 1);
	IplImage* pyr = cvCreateImage(cvSize(sz.width / 2, sz.height / 2), 8, 3);
	IplImage* tgray;
	CvSeq* result;
	double s, t, n, m;
	CvSeq* squares = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvPoint), storage);
	cvSetImageROI(timg, cvRect(0, 0, sz.width, sz.height));
	cvPyrDown(timg, pyr, 7);
	cvPyrUp(pyr, timg, 7);
	tgray = cvCreateImage(sz, 8, 1);
	cnt = 0;
	for (c = 0; c < 3; c++)
	{
		cvSetImageCOI(timg, c + 1);
		cvCopy(timg, tgray, 0);
		cvCanny(tgray, gray, 0, thresh, 5);
		cvDilate(gray, gray, 0, 1);
		cvFindContours(gray, storage, &contours, sizeof(CvContour),
			CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
		while (contours)
		{
			cnt += 1;
			contours = contours->h_next;
		}
	}
	cvReleaseImage(&gray);
	cvReleaseImage(&pyr);
	cvReleaseImage(&tgray);
	cvReleaseImage(&timg);
	cvClearMemStorage(storage);
	return cnt/3;

}
//this function is used for finding squares in the picture
CvSeq* findSquares4(IplImage* img, CvMemStorage* storage, int* numbers)
{
	CvSeq* contours;
	int i, c;
	CvSize sz = cvSize(img->width & -2, img->height & -2);

	IplImage* timg = cvCloneImage(img);
	IplImage* gray = cvCreateImage(sz, 8, 1);
	IplImage* pyr = cvCreateImage(cvSize(sz.width / 2, sz.height / 2), 8, 3);
	IplImage* tgray;
	CvSeq* result;
	double s, t, n, m;
	CvSeq* squares = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvPoint), storage);
	cvSetImageROI(timg, cvRect(0, 0, sz.width, sz.height));
	cvPyrDown(timg, pyr, 7);
	cvPyrUp(pyr, timg, 7);
	tgray = cvCreateImage(sz, 8, 1);
	*numbers = 0;
	for (c = 0; c < 3; c++)
	{
		cvSetImageCOI(timg, c + 1);
		cvCopy(timg, tgray, 0); 
			
				cvCanny(tgray, gray, 0, thresh, 5);
				cvDilate(gray, gray, 0, 1);
			cvFindContours(gray, storage, &contours, sizeof(CvContour),
				CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
			while (contours)
			{
				result = cvApproxPoly(contours, sizeof(CvContour), storage,
					CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0);

				if (result->total == 4 &&
					fabs(cvContourArea(result, CV_WHOLE_SEQ)) > 500 &&
					fabs(cvContourArea(result, CV_WHOLE_SEQ)) < 1000000 &&
					cvCheckContourConvexity(result))
				{
					s = 0;
					for (i = 0; i < 5; i++)
					{
						if (i >= 2)
						{
							t = fabs(angle(
								(CvPoint*)cvGetSeqElem(result, i),
								(CvPoint*)cvGetSeqElem(result, i - 2),
								(CvPoint*)cvGetSeqElem(result, i - 1)));
							s = s > t ? s : t;
						}
					}
					m = 0;
					for (i = 0; i < 5; i++)
					{
						if (i >= 2)
						{
							n = distance(
								(CvPoint*)cvGetSeqElem(result, i),
								(CvPoint*)cvGetSeqElem(result, i - 2),
								(CvPoint*)cvGetSeqElem(result, i - 1));
							m = n == 1 ? n : m;
						}
					}
					if (s < 0.03)
					{
						if (m == 1)
						{
							*numbers += 1;
							for (i = 0; i < 4; i++)
								cvSeqPush(squares,
								(CvPoint*)cvGetSeqElem(result, i));
						}
					}
					
						


				}
				contours = contours->h_next;
			}
		}
	cvReleaseImage(&gray);
	cvReleaseImage(&pyr);
	cvReleaseImage(&tgray);
	cvReleaseImage(&timg);
	cvClearMemStorage(storage);
	*numbers = *numbers / 3;
	return squares;
}
//this function is used for detecting diamons, where img indicate the input image, storage indicate the storage buffer, numbers indicate the number of the dimonds in the input image.
CvSeq* findDiamond(IplImage* img, CvMemStorage* storage, int* numbers)
{
	CvSeq* contours;
	int i, c;
	CvSize sz = cvSize(img->width & -2, img->height & -2);
	IplImage* timg = cvCloneImage(img);
	IplImage* gray = cvCreateImage(sz, 8, 1);
	IplImage* pyr = cvCreateImage(cvSize(sz.width / 2, sz.height / 2), 8, 3);
	IplImage* tgray;
	CvSeq* result;
	double s, t, n, m;
	CvSeq* diamonds = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvPoint), storage);
	cvSetImageROI(timg, cvRect(0, 0, sz.width, sz.height));
	cvPyrDown(timg, pyr, 7);
	cvPyrUp(pyr, timg, 7);
	tgray = cvCreateImage(sz, 8, 1);
	*numbers = 0;
	for (c = 0; c < 3; c++)
	{
		cvSetImageCOI(timg, c + 1);
		cvCopy(timg, tgray, 0);
		cvCanny(tgray, gray, 0, thresh, 5);
		cvDilate(gray, gray, 0, 1);
		cvFindContours(gray, storage, &contours, sizeof(CvContour),
			CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
		while (contours)
		{
			result = cvApproxPoly(contours, sizeof(CvContour), storage,
				CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0);
			if (result->total == 4 &&
				fabs(cvContourArea(result, CV_WHOLE_SEQ)) > 500 &&
				fabs(cvContourArea(result, CV_WHOLE_SEQ)) < 1000000 &&
				cvCheckContourConvexity(result))
			{
				s = 0;
				for (i = 0; i < 5; i++)
				{
					// find angle between joint edges (maximum of cosine)
					if (i >= 2)
					{
						t = fabs(angle(
							(CvPoint*)cvGetSeqElem(result, i),
							(CvPoint*)cvGetSeqElem(result, i - 2),
							(CvPoint*)cvGetSeqElem(result, i - 1)));
						s = s > t ? s : t;
					}
				}
				//check the contours by detecting the cosine of the angles and 
				//the length of the side to decide whether the contour is a diamond
				m = 0;
				for (i = 0; i < 5; i++)
				{
					if (i >= 2)
					{
						n = distance(
							(CvPoint*)cvGetSeqElem(result, i),
							(CvPoint*)cvGetSeqElem(result, i - 2),
							(CvPoint*)cvGetSeqElem(result, i - 1));
						m = n == 1 ? n : m;
					}
				}
				if (s > 0.03)
				{
					if (m == 1)
					{
						*numbers += 1;
						for (i = 0; i < 4; i++)
							cvSeqPush(diamonds,
							(CvPoint*)cvGetSeqElem(result, i));
					}
				}

			}
			contours = contours->h_next;
		}
	}
	cvReleaseImage(&gray);
	cvReleaseImage(&pyr);
	cvReleaseImage(&tgray);
	cvReleaseImage(&timg);
	cvClearMemStorage(storage);
	*numbers = *numbers/3;
	return diamonds;
}

//the same as former, used for detecting the triangles
CvSeq* findTriangle(IplImage* img, CvMemStorage* storage, int* numbers)
{
	CvSeq* contours;
	int i, c;
	CvSize sz = cvSize(img->width & -2, img->height & -2);

	IplImage* timg = cvCloneImage(img);
	IplImage* gray = cvCreateImage(sz, 8, 1);
	IplImage* pyr = cvCreateImage(cvSize(sz.width / 2, sz.height / 2), 8, 3);
	IplImage* tgray;
	CvSeq* result;
	double s, t, n, m;
	CvSeq* triangles = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvPoint), storage);
	cvSetImageROI(timg, cvRect(0, 0, sz.width, sz.height));
	cvPyrDown(timg, pyr, 7);
	cvPyrUp(pyr, timg, 7);
	tgray = cvCreateImage(sz, 8, 1);
	*numbers = 0;
	for (c = 0; c < 3; c++)
	{
		cvSetImageCOI(timg, c + 1);
		cvCopy(timg, tgray, 0);
		cvCanny(tgray, gray, 0, thresh, 5);
		cvDilate(gray, gray, 0, 1);
		cvFindContours(gray, storage, &contours, sizeof(CvContour),
			CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
		while (contours)
		{
			result = cvApproxPoly(contours, sizeof(CvContour), storage,
				CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0);
			if (result->total == 3 &&
				fabs(cvContourArea(result, CV_WHOLE_SEQ)) > 500 &&
				fabs(cvContourArea(result, CV_WHOLE_SEQ)) < 1000000 &&
				cvCheckContourConvexity(result))
			{
				*numbers += 1;
				for (i = 0; i < 3; i++)
					cvSeqPush(triangles,
					(CvPoint*)cvGetSeqElem(result, i));
			}
			contours = contours->h_next;
		}
	}
	cvReleaseImage(&gray);
	cvReleaseImage(&pyr);
	cvReleaseImage(&tgray);
	cvReleaseImage(&timg);
	cvClearMemStorage(storage);
	*numbers = *numbers / 3;
	return triangles;
}

//detecting rectangles
CvSeq* findrectangles4(IplImage* img, CvMemStorage* storage, int* numbers )

{
	CvSeq* contours;
	int i, c;
	CvSize sz = cvSize(img->width & -2, img->height & -2);

	IplImage* timg = cvCloneImage(img);
	IplImage* gray = cvCreateImage(sz, 8, 1);
	IplImage* pyr = cvCreateImage(cvSize(sz.width / 2, sz.height / 2), 8, 3);
	IplImage* tgray;
	CvSeq* result;
	double s, t, n, m;
	CvSeq* rectangles = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvPoint), storage);
	cvSetImageROI(timg, cvRect(0, 0, sz.width, sz.height));
	// filter the noise
	cvPyrDown(timg, pyr, 7);
	cvPyrUp(pyr, timg, 7);
	tgray = cvCreateImage(sz, 8, 1);
	*numbers = 0;
	//process the picture in one elemental color
	for (c = 0; c < 3; c++)
	{
		cvSetImageCOI(timg, c + 1);
		cvCopy(timg, tgray, 0);
			// apply canny and set the upper threshold 
				cvCanny(tgray, gray, 0, thresh, 5);
				cvDilate(gray, gray, 0, 1);
			cvFindContours(gray, storage, &contours, sizeof(CvContour),
				CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
			while (contours)
			{
				//approximate the contours with polygons;
				result = cvApproxPoly(contours, sizeof(CvContour), storage,
					CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0);

				if (result->total == 4 &&
					fabs(cvContourArea(result, CV_WHOLE_SEQ)) > 500 &&
					fabs(cvContourArea(result, CV_WHOLE_SEQ)) < 1000000 &&
					//detect the convexity
					cvCheckContourConvexity(result))
				{
					s = 0;
					for (i = 0; i < 5; i++)
					{
						// find minimum angle between joint edges (maximum of cosine)
						if (i >= 2)
						{
							t = fabs(angle(
								(CvPoint*)cvGetSeqElem(result, i),
								(CvPoint*)cvGetSeqElem(result, i - 2),
								(CvPoint*)cvGetSeqElem(result, i - 1)));
							s = s > t ? s : t;
						}
					}
					//detect the length of the side and the degree of the angles
					m = 0;
					for (i = 0; i < 5; i++)
					{
						if (i >= 2)
						{
							n = distance(
								(CvPoint*)cvGetSeqElem(result, i),
								(CvPoint*)cvGetSeqElem(result, i - 2),
								(CvPoint*)cvGetSeqElem(result, i - 1));
							m = n == 1 ? n : m;
						}
					}
					if (s < 0.08)
					{
						if (m == 0)
						{
							*numbers += 1;
							for (i = 0; i < 4; i++)
								cvSeqPush(rectangles,
								(CvPoint*)cvGetSeqElem(result, i));
						}
					}




				}
				contours = contours->h_next;
			}
		}
	cvReleaseImage(&gray);
	cvReleaseImage(&pyr);
	cvReleaseImage(&tgray);
	cvReleaseImage(&timg);
	cvClearMemStorage(storage);
	*numbers = *numbers / 3;
	return rectangles;
}
//The following function should be used to detect polygon
//img_num should be the integer that used to describe the picture's number
CvSeq* findpolygons(IplImage* img, CvMemStorage* storage, int* numbers, int img_num)
{
	CvSeq* contours;
	int i, c;
	for (i = 0; i < 10; i++)
	{
		polygon[img_num][i][1] = 1;
	}
	CvSize sz = cvSize(img->width & -2, img->height & -2);

	IplImage* timg = cvCloneImage(img);
	IplImage* gray = cvCreateImage(sz, 8, 1);
	IplImage* pyr = cvCreateImage(cvSize(sz.width / 2, sz.height / 2), 8, 3);
	IplImage* tgray;
	CvSeq* result;
	double s, t, n, m;
	//create a empty sequence to store the point of the contours
	CvSeq* squares = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvPoint), storage);
	cvSetImageROI(timg, cvRect(0, 0, sz.width, sz.height));
	//filter the image
	cvPyrDown(timg, pyr, 7);
	cvPyrUp(pyr, timg, 7);
	tgray = cvCreateImage(sz, 8, 1);
	//used to store the poly
	i = 0;
	//used to store the convexity
	//int conv = 1;
	//used to store the numbers
	//int numbers = 0;
	*numbers = 0;

		// find one specific color
		cvSetImageCOI(timg, 1);
		cvCopy(timg, tgray, 0);
		// apply Canny. Take the upper threshold from slider
		// Canny helps to catch squares with gradient shading  
		cvCanny(tgray, gray, 0, thresh, 5);
		//dilate the contoiurs
		cvDilate(gray, gray, 0, 1);
		//find each contours
		cvFindContours(gray, storage, &contours, sizeof(CvContour),
			CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
		//ergodic every contours
		while (contours)
		{
			//approximate the contours with polygons
			result = cvApproxPoly(contours, sizeof(CvContour), storage,
				CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0);

			if (result->total != 4 && result->total != 3)
			{
				i = result->total;
				polygon[img_num][i - 1][0] += 1;
				*numbers += 1;
				//check the convexity
				if (!cvCheckContourConvexity(result))
				{
					polygon[img_num][i - 1][1] = 0;
				}
			}
			// go to next contours
			contours = contours->h_next;
		}
	
	cvReleaseImage(&gray);
	cvReleaseImage(&pyr);
	cvReleaseImage(&tgray);
	cvReleaseImage(&timg);
	cvClearMemStorage(storage);
	//*numbers = *numbers / 3;
	//cvClearSeq(contours);
	return squares;
}

//This function is used to detect the circle
vector<Vec3f> findCircle(Mat img, CvMemStorage* storage, int* numbers)
{
	*numbers = 0;
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	GaussianBlur(gray, gray, Size(9, 9), 2, 2);

	vector<Vec3f> circles;
	//Hough algorithm 
	HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 1, gray.rows / 20, 100, 60, 0, 0);
	*numbers = circles.size();
	return circles;

}

//function drawShapes is used for drawing all the figures in the picture
void drawShapes(IplImage* img, CvSeq* squares, CvSeq* rectangles, vector<Vec3f> circles, CvSeq* diamonds, CvSeq* triangles)
{
	CvSeqReader reader;
	CvSeqReader reader2;
	CvSeqReader reader3;
	CvSeqReader reader4;
	IplImage* cpy = cvCloneImage(img);
	int i;
	cvStartReadSeq(squares, &reader, 0);
	cvStartReadSeq(rectangles, &reader2, 0);
	cvStartReadSeq(diamonds, &reader3, 0);
	cvStartReadSeq(triangles, &reader4, 0);
	// read 4 sequence elements at a time (all vertices of a square)
	for (i = 0; i < squares->total; i += 4)
	{
		CvPoint pt[4], *rect = pt;
		int count = 4;
		// read 4 vertices
		CV_READ_SEQ_ELEM(pt[0], reader);
		CV_READ_SEQ_ELEM(pt[1], reader);
		CV_READ_SEQ_ELEM(pt[2], reader);
		CV_READ_SEQ_ELEM(pt[3], reader);
		// draw the square as a closed polyline
		cvPolyLine(cpy, &rect, &count, 1, 1, CV_RGB(0, 255, 0), 2, CV_AA, 0);
	}
	for (i = 0; i < rectangles->total; i += 4)
	{
		CvPoint pt2[4], *rect2 = pt2;
		int count2 = 4;
		// read 4 vertices
		CV_READ_SEQ_ELEM(pt2[0], reader2);
		CV_READ_SEQ_ELEM(pt2[1], reader2);
		CV_READ_SEQ_ELEM(pt2[2], reader2);
		CV_READ_SEQ_ELEM(pt2[3], reader2);
		// draw the square as a closed polyline
		cvPolyLine(cpy, &rect2, &count2, 1, 1, CV_RGB(255, 0, 0), 2, CV_AA, 0);
	}
	for (i = 0; i < triangles->total; i += 3)
	{
		CvPoint pt4[3], *rect4 = pt4;
		int count4 = 3;
		//read 4 vertices
		CV_READ_SEQ_ELEM(pt4[0], reader4);
		CV_READ_SEQ_ELEM(pt4[1], reader4);
		CV_READ_SEQ_ELEM(pt4[2], reader4);
		//CV_READ_SEQ_ELEM(pt3[3], reader3);
		// draw the square as a closed polyline
		cvPolyLine(cpy, &rect4, &count4, 1, 1, CV_RGB(249, 204, 226), 2, CV_AA, 0);

	}
	for (i = 0; i < diamonds->total; i += 4)
	{
		CvPoint pt3[4], *rect3 = pt3;
		int count3 = 4;
		//read 4 vertices
		CV_READ_SEQ_ELEM(pt3[0], reader3);
		CV_READ_SEQ_ELEM(pt3[1], reader3);
		CV_READ_SEQ_ELEM(pt3[2], reader3);
		CV_READ_SEQ_ELEM(pt3[3], reader3);
		// draw the square as a closed polyline
		cvPolyLine(cpy, &rect3, &count3, 1, 1, CV_RGB(0, 0, 255), 2, CV_AA, 0);

	}
	for (size_t i = 0; i < circles.size(); i++)
	{
		//extract the center point of the circle
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		//extract the radius of the circle
		int radius = cvRound(circles[i][2]);
		cvCircle(cpy,
			CvPoint(center),//center point 
			cvRound(radius), //radius
			CV_RGB(254, 254, 65),
			6);
	}
	cvShowImage(wndname, cpy);
	cvReleaseImage(&cpy);
	cvClearSeq(rectangles);
	cvClearSeq(squares);
}

//this function is used to find the different characteristic between four of them
int find_different(int a, int b, int c, int d)
{
	//if they are the same, return 0;
	if (a == b && a == c && a == d)
	{
		return 0;
	}
	//if three of them are the same, return the different one
	else if (a == b && a == c)
	{
		//4 is the different
		return 4;
	}
	else if (a == b && a == d)
	{
		//c is the different
		return 3;
	}
	else if (a == c && a == d)
	{
		//b is different
		return 2;
	}
	else if (b == c && b == d)
	{
		//a is different
		return 1;
	}
	else
	{
		int ave = (a + b + c + d) / 4;
		int a0 = fabs(a - ave);
		int b0 = fabs(b - ave);
		int c0 = fabs(c - ave);
		int d0 = fabs(d - ave);
		//sort a0, b0, c0, d0, reutrn the biggest one.
		int unsorted[] = { a0, b0, c0, d0 };
		for (int i = 0; i < sizeof(unsorted) / sizeof(int); i++)
		{
			for (int j = i; j < sizeof(unsorted) / sizeof(int); j++)
			{
				if (unsorted[i] > unsorted[j])
				{
					int temp = unsorted[i];
					unsorted[i] = unsorted[j];
					unsorted[j] = temp;
				}
			}
		}
		if (unsorted[3] == a0)
		{
			return 1;
		}
		if (unsorted[3] == b0)
		{
			return 2;
		}
		if (unsorted[3] == c0)
		{
			return 3;
		}
		if (unsorted[3] == d0)
		{
			return 4;
		}
	}
}

int main(int argc, char** argv)
{
	int i, c;
	storage0= cvCreateMemStorage(0);
	storage = cvCreateMemStorage(0);
	storage2 = cvCreateMemStorage(0);
	storage3 = cvCreateMemStorage(0);
	storage4 = cvCreateMemStorage(0);
	storage5 = cvCreateMemStorage(0);
	storage6 = cvCreateMemStorage(0);
	//this array is used to store the characteristics of the shapes,
	//the first variable is used to describe the number of the picture, 
	//the second variable is used to describe the characteristcs, which is
	//(the number of contours,squares,diamonds,rectangules,circles,polygons,triangles)
	int characteristic[4][7];
	

	string controller;
	string name1;
	string name2;
	string name3;
	string name4;
	cout << "do you want to continue the process?(Y/N)" << endl;
	getline(cin, controller);
	while (controller == "Y")
	{
		cout << "input the name of the picture:" << endl;
		getline(cin, name1);
		if (name1 != "exit")
		{

			char* names = (char *)name1.data();
			img0 = cvLoadImage(names, 1);
			while (img0 == 0)
			{
				cout << "can't load picture:" << names 
					<< " please input the name again:" << endl;
				getline(cin, name1);
				if (name1 == "exit")
				{
					return 0;
				}
				names = (char*)name1.data();
				img0 = cvLoadImage(names, 1);
			}
			//the following part is to process the first picture
			if (img = cvCloneImage(img0))
				cvNamedWindow(wndname, 1);
			//cnt is used to store the number of the contours
			characteristic[0][0] = count_the_contours(img, storage0);
			// find and draw the squares
			CvSeq* square = findSquares4(img, storage, &characteristic[0][1]);
			CvSeq* diamonds = findDiamond(img, storage4, &characteristic[0][2]);
			CvSeq* rectangles = findrectangles4(img, storage2, &characteristic[0][3]);
			CvSeq* polygons = findpolygons(img, storage5, &characteristic[0][5], 0);
			CvSeq* triangles = findTriangle(img, storage6, &characteristic[0][6]);
			//Cause HoughCircle algorthm need another kind of format of the image, so I read the image in that data structure for one more time.
			Mat src = imread(names);
			if (!src.data)
				return -1;
			vector<Vec3f> Circles = findCircle(src, storage3, &characteristic[0][4]);
			cout << "in the picture " << name1 << ":" << endl;
			cout << "number of contours is:" << characteristic[0][0] << endl;
			cout << "number of squares is:" << characteristic[0][1] << endl;
			cout << "number of daimonds is:" << characteristic[0][2] << endl;
			cout << "number of rectangles is:" << characteristic[0][3] << endl;
			cout << "number of circles is:" << characteristic[0][4] << endl;
			cout << "number of triangles is:" << characteristic[0][6] << endl;
			cout << "number of polygons is:" << characteristic[0][5] - characteristic[0][4] << endl;
			drawShapes(img, square, rectangles, Circles, diamonds, triangles);
			c = cvWaitKey(-1);
			cvReleaseImage(&img);
			cvReleaseImage(&img0);
			cvClearMemStorage(storage0);
			cvClearMemStorage(storage);
			cvClearMemStorage(storage2);
			cvClearMemStorage(storage3);
			cvClearMemStorage(storage4);
			cvClearMemStorage(storage5);
			cvClearMemStorage(storage6);
			cvClearSeq(rectangles);
			cvClearSeq(square);
			//this part is for the second picture;
			cout << "input the name of the second picture" << endl;
			getline(cin, name2);
			if (name2 != "exit")
			{
				char* names2 = (char *)name2.data();
				img2_0 = cvLoadImage(names2, 1);
				while (img2_0 == 0)
				{
					cout << "can't load picture:" << names2 
						<< " please input the name again:" << endl;
					getline(cin, name2);
					if (name2 == "exit")
					{
						return 0;
					}
					names2 = (char *)name2.data();
					img2_0 = cvLoadImage(names2, 1);
				}

				
				//the following part is to process the second part
				if (img2 = cvCloneImage(img2_0))
					cvNamedWindow(wndname, 1);
				characteristic[1][0] = count_the_contours(img2, storage0);
				CvSeq* square2 = findSquares4(img2, storage, &characteristic[1][1]);
				CvSeq* diamonds2 = findDiamond(img2, storage4, &characteristic[1][2]);
				CvSeq* rectangles2 = findrectangles4(img2, storage2, &characteristic[1][3]);
				CvSeq* polygons2 = findpolygons(img2, storage5, &characteristic[1][5], 1);
				CvSeq* triangles2 = findTriangle(img2, storage6, &characteristic[1][6]);
				//Because HoughCircle algorthm need another kind of format of the image, so I read the image in that data structure for one more time.
				Mat src2 = imread(names2);
				if (!src2.data)
					return -1;
				vector<Vec3f> Circles2 = findCircle(src2, storage3, &characteristic[1][4]);
				cout << "" << endl;
				cout << "in the picture " << name2 << ":" << endl;
				cout << "number of contours is:" << characteristic[1][0] << endl;
				cout << "number of squares is:" << characteristic[1][1] << endl;
				cout << "number of daimonds is:" << characteristic[1][2] << endl;
				cout << "number of rectangles is:" << characteristic[1][3] << endl;
				cout << "number of circles is:" << characteristic[1][4] << endl;
				cout << "number of triangles is:" << characteristic[1][6] << endl;
				cout << "number of polygons is:" << characteristic[1][5] - characteristic[1][4] << endl;
				drawShapes(img2, square2, rectangles2, Circles2, diamonds2, triangles2);
				c = cvWaitKey(-1);


				cvReleaseImage(&img2);
				cvReleaseImage(&img2_0);
				cvClearMemStorage(storage0);
				cvClearMemStorage(storage);
				cvClearMemStorage(storage2);
				cvClearMemStorage(storage3);
				cvClearMemStorage(storage4);
				cvClearMemStorage(storage5);
				cvClearMemStorage(storage6);
				cvClearSeq(rectangles2);
				cvClearSeq(square2);

				//I should implement the logic to read the third image.
				cout << "input the name of the third picture:" << endl;
				getline(cin, name3);
				char* names3 = (char*)name3.data();
				img3_0 = cvLoadImage(names3, 1);
				while (img3_0 == 0)
				{
					cout << "can't load picture:" << names3
						<< " please input the name again:" << endl;
					getline(cin, name3);
					if (name3 == "exit")
					{
						return 0;
					}
					names3 = (char *)name3.data();
					img3_0 = cvLoadImage(names3, 1);
				}
				if (img3 = cvCloneImage(img3_0))
					cvNamedWindow(wndname, 1);
				characteristic[2][0] = count_the_contours(img3, storage0);
				CvSeq* square3 = findSquares4(img3, storage, &characteristic[2][1]);
				CvSeq* diamonds3 = findDiamond(img3, storage4, &characteristic[2][2]);
				CvSeq* rectangles3 = findrectangles4(img3, storage2, &characteristic[2][3]);
				CvSeq* polygons3 = findpolygons(img3, storage5, &characteristic[2][5], 2);
				CvSeq* triangles3 = findTriangle(img3, storage6, &characteristic[2][6]);
				Mat src3 = imread(names3);
				if (!src3.data)
					return -1;
				vector<Vec3f> Circles3 = findCircle(src3, storage3, &characteristic[2][4]);
				cout << "" << endl;
				cout << "in the picture " << name3 << ":" << endl;
				cout << "number of contours is:" << characteristic[2][0] << endl;
				cout << "number of squares is:" << characteristic[2][1] << endl;
				cout << "number of daimonds is:" << characteristic[2][2] << endl;
				cout << "number of rectangles is:" << characteristic[2][3] << endl;
				cout << "number of circles is:" << characteristic[2][4] << endl;
				cout << "number of triangles is:" << characteristic[2][6] << endl;
				cout << "number of polygons is:" << characteristic[2][5] - characteristic[2][4] << endl;
				drawShapes(img3, square3, rectangles3, Circles3, diamonds3, triangles3);
				c = cvWaitKey(-1);


				cvReleaseImage(&img3);
				cvReleaseImage(&img3_0);
				cvClearMemStorage(storage0);
				cvClearMemStorage(storage);
				cvClearMemStorage(storage2);
				cvClearMemStorage(storage3);
				cvClearMemStorage(storage4);
				cvClearMemStorage(storage5);
				cvClearMemStorage(storage6);
				cvClearSeq(rectangles3);
				cvClearSeq(square3);

				//the following code is to read the fourth picture
				cout << "input the name of the fourth picture:" << endl;
				getline(cin, name4);
				char* names4 = (char*)name4.data();
				img4_0 = cvLoadImage(names4, 1);
				while (img4_0 == 0)
				{
					cout << "can't load picture:" << names4
						<< " please input the name again:" << endl;
					getline(cin, name4);
					if (name4 == "exit")
					{
						return 0;
					}
					names4 = (char *)name4.data();
					img4_0 = cvLoadImage(names4, 1);
				}
				if (img4 = cvCloneImage(img4_0))
					cvNamedWindow(wndname, 1);
				characteristic[3][0] = count_the_contours(img4, storage0);
				CvSeq* square4 = findSquares4(img4, storage, &characteristic[3][1]);
				CvSeq* diamonds4 = findDiamond(img4, storage4, &characteristic[3][2]);
				CvSeq* rectangles4 = findrectangles4(img4, storage2, &characteristic[3][3]);
				CvSeq* polygons4 = findpolygons(img4, storage5, &characteristic[3][5], 3);
				CvSeq* triangles4 = findTriangle(img4, storage6, &characteristic[3][6]);
				Mat src4 = imread(names4);
				if (!src4.data)
					return -1;
				vector<Vec3f> Circles4 = findCircle(src4, storage3, &characteristic[3][4]);
				cout << "" << endl;
				cout << "in the picture " << name4 << ":" << endl;
				cout << "number of contours is:" << characteristic[3][0] << endl;
				cout << "number of squares is:" << characteristic[3][1] << endl;
				cout << "number of daimonds is:" << characteristic[3][2] << endl;
				cout << "number of rectangles is:" << characteristic[3][3] << endl;
				cout << "number of circles is:" << characteristic[3][4] << endl;
				cout << "number of triangles is:" << characteristic[3][6] << endl;
				cout << "number of polygons is:" << characteristic[3][5] - characteristic[3][4] << endl;
				drawShapes(img4, square4, rectangles4, Circles4, diamonds4, triangles4);
				c = cvWaitKey(-1);


				cvReleaseImage(&img4);
				cvReleaseImage(&img4_0);
				cvClearMemStorage(storage0);
				cvClearMemStorage(storage);
				cvClearMemStorage(storage2);
				cvClearMemStorage(storage3);
				cvClearMemStorage(storage4);
				cvClearMemStorage(storage5);
				cvClearMemStorage(storage6);
				cvClearSeq(rectangles4);
				cvClearSeq(square4);
				//the comparation part should be implemented here
				//check the number of the contours, which has the highest weight.
				//a, b, c, d is used to calculate the "wired degree", the picture with the highest "wired degree" would be considered as the most wired picture.
				int a, b, c, d;
				a = 0;
				b = 0;
				c = 0;
				d = 0;
				//i is used to receive the result of the function find_different;
				int i, j;
				i = find_different(characteristic[0][0], characteristic[1][0], characteristic[2][0], characteristic[3][0]);
				switch (i)
				{
				    case 0:
						break;
					case 1:
						a += 1000;
						break;
					case 2:
						b += 1000;
						break;
					case 3:
						c += 1000;
						break;
					case 4:
						d += 1000;
						break;
				}
				//check the convexity of the picture, it has the second high weight
				for (i = 0; i < 10; i++)
				{
					j = find_different(polygon[0][i][1], polygon[1][i][1], polygon[2][i][1], polygon[3][i][1]);
					switch (j)
					{
					case 0:
						break;
					case 1:
						a += 200;
						break;
					case 2:
						b += 200;
						break;
					case 3:
						c += 200;
						break;
					case 4:
						d += 200;
						break;
					}
				}

				//check the number of the basic shapes(or polygons), which is the number of contours minus number of polygons(the number of polygons contains circles, because circles would be recognized as polygongs)
				//this characteristic has the third high weight
				for (i = 0; i < 4; i++)
				{
					characteristic[i][5] = characteristic[i][5] - characteristic[i][4];
				}
				i = find_different(characteristic[0][5], characteristic[1][5], characteristic[2][5], characteristic[3][5]);
				switch(i)
				{
					case 0:
						break;
					case 1:
						a += 100;
						break;
					case 2:
						b += 100;
						break;
					case 3:
						c += 100;
						break;
					case 4:
						d += 100;
						break;
				}
				//check the number of each specific basic shape, has the third high weight
				for (i = 1; i < 5; i++)
				{
					j = find_different(characteristic[0][i], characteristic[1][i], characteristic[2][i], characteristic[3][i]);
					switch (j)
					{
					case 0:
						break;
					case 1:
						a += 10;
						break;
					case 2:
						b += 10;
						break;
					case 3:
						c += 10;
						break;
					case 4:
						d += 10;
						break;
					}
				}
				//this is for triangles checking, which is characteristic[][6]
				j = find_different(characteristic[0][6], characteristic[1][6], characteristic[2][6], characteristic[3][6]);
				switch (j)
				{
				case 0:
					break;
				case 1:
					a += 10;
					break;
				case 2:
					b += 10;
					break;
				case 3:
					c += 10;
					break;
				case 4:
					d += 10;
					break;
				}
				//sort the a,b,c,d
				cout << "" << endl;
				if (a == b&&a == c&&a == d)
				{
					cout << "sorry I cannot find any different between them" << endl;
				}
				else
				{
					int unsorted[] = { a,b,c,d };
					for (int i = 0; i < sizeof(unsorted) / sizeof(int); i++)
					{
						for (int j = i; j < sizeof(unsorted) / sizeof(int); j++)
						{
							if (unsorted[i] > unsorted[j])
							{
								int temp = unsorted[i];
								unsorted[i] = unsorted[j];
								unsorted[j] = temp;
							}
						}
					}
					if (unsorted[3] == a)
						cout << "picture 1:" << name1 << " is the odd one" << endl;
					if (unsorted[3] == b)
						cout << "picture 2:" << name2 << " is the odd one" << endl;
					if (unsorted[3] == c)
						cout << "picture 3:" << name3 << " is the odd one" << endl;
					if (unsorted[3] == d)
						cout << "picture 4:" << name4 << " is the odd one" << endl;

				}	
				cout << "" << endl;
				memset(polygon, 0, sizeof(polygon));
				memset(characteristic, 0, sizeof(characteristic));
				if ((char)c == 27)
					break;
			}
			else
			{
				return 0;
			}
		}
		else 
		{
			return 0;
		}
		cout << "do you want to continue the process?(Y/N)" << endl;
		getline(cin, controller);
	}
	cvDestroyWindow(wndname);
	return 0;
}