/**
 * Created by crou on 26.03.15.
 */


import javafx.util.converter.PercentageStringConverter;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;

import java.awt.*;
import java.lang.management.MonitorInfo;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_highgui.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

public class Main {

    private static boolean calibrated;
    private static int hueMin = 0;
    private static int hueMax = 180;
    private static int valueMin = 0;
    private static int valueMax = 255;
    private static int satMin = 0;
    private static int satMax = 255;
    private static int yMax = 255;
    private static int yMin = 0;
    private static int CrMax = 255;
    private static int CrMin = 0;
    private static int CbMax = 255;
    private static int CbMin = 0;

    private static CvCapture camera;
    private static IplImage original = cvCreateImage(CvSize.ZERO, IPL_DEPTH_8U, 1);
    private static IplImage mask;

    private static byte[] calibrationHueAndValue;

    public static void main(String[] args) {

        String originalWindow = "original window";

        camera = cvCreateCameraCapture(0);
        cvSetCaptureProperty(camera, CV_CAP_PROP_GAMMA, 0.300);

        IplConvKernel kernel = cvCreateStructuringElementEx(3, 3, 1, 1, CV_SHAPE_ELLIPSE);
        IplConvKernel kernel2 = cvCreateStructuringElementEx(7, 7, 4, 4, CV_SHAPE_ELLIPSE);

        IplImage patern = cvQueryFrame(camera);

        if (camera != null) {
            cvNamedWindow(originalWindow, 0);
            cvNamedWindow("Value", 0);
            cvNamedWindow("Hue", 0);
            cvNamedWindow("Saturation", 0);

            CvMemStorage memory = cvCreateMemStorage(0xffff);
            CvSeq contour = new CvContour();

            for (;;) {

                original.release();
                original = cvQueryFrame(camera);

                mask = cvCreateImage(original.cvSize(), IPL_DEPTH_8U, 1);
                IplImage h = cvCreateImage(original.cvSize(), IPL_DEPTH_8U, 1);
                IplImage v  = cvCreateImage(original.cvSize(), IPL_DEPTH_8U, 1);
                IplImage s  = cvCreateImage(original.cvSize(), IPL_DEPTH_8U, 1);

                cvCvtColor(original, original, CV_BGR2HSV);
                cvSplit(original, h, s, v, null);

                cvInRangeS(h, cvScalar(hueMin), cvScalar(hueMax), h);
                cvInRangeS(v, cvScalar(valueMin), cvScalar(valueMax), v);
                cvInRangeS(s, cvScalar(satMin), cvScalar(satMax), s);

                cvSmooth(h, h, CV_GAUSSIAN, 3, 3, 2, 2);
                cvThreshold(h, h, 100, 255, CV_THRESH_BINARY);
                cvMorphologyEx(h, h, null, kernel, CV_MOP_OPEN, 2);

                cvAnd(h, s, mask);
                cvAnd(mask, v, mask);

                cvSmooth(mask, mask, CV_GAUSSIAN, 3, 3, 2, 2);
                cvThreshold(mask, mask, 10, 255, CV_THRESH_BINARY_INV);

                cvCvtColor(original, original, CV_HSV2BGR);

                cvFindContours(mask, memory, contour, Loader.sizeof(CvContour.class), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

                CvSeq theBigestContour = contour;

                while (contour != null && !contour.isNull()) {
                    if (contour.total() > 0) {
                        CvSize2D32f size = cvMinAreaRect2(theBigestContour, memory).size();
                        CvSize2D32f curSize = cvMinAreaRect2(contour, memory).size();

                        if (size.height() * size.width() < curSize.height() * curSize.width()) {
                            theBigestContour = contour;
                        }
                    }
                    contour = contour.h_next();
                }

                contour = theBigestContour;

                if (theBigestContour != null && !theBigestContour.isNull()) {

                    CvSeq approxy = cvApproxPoly(theBigestContour, Loader.sizeof(CvContour.class), memory, CV_POLY_APPROX_DP, 2, 1);


                    if(approxy != null && !approxy.isNull()) {
                            CvPoint p0 = new CvPoint(cvGetSeqElem(approxy, approxy.total() - 1));
                            for (int i = 0; i < approxy.total(); i++) {
                                BytePointer pointer = cvGetSeqElem(approxy, i);
                                CvPoint p = new CvPoint(pointer);
                                cvLine(original, p0, p, CvScalar.GREEN, 2, 8, 0);
                                p0 = p;
                        }

                        CvSeq convexHull = cvConvexHull2(approxy, memory, CV_CLOCKWISE, 0);
                        CvSeq convexHullForDrawing = cvConvexHull2(approxy, memory, CV_CLOCKWISE, 1);

                        if(convexHullForDrawing != null && !convexHullForDrawing.isNull() && convexHullForDrawing.total() > 0) {
                            CvPoint pt0 = new CvPoint(cvGetSeqElem(convexHullForDrawing, convexHullForDrawing.total() - 1));
                            for (int i = 0; i < convexHullForDrawing.total(); i++) {

                                CvPoint pt = new CvPoint(cvGetSeqElem(convexHullForDrawing, i));

                            }
                        }

                        CvMoments moments = new CvMoments();

                        cvMoments(theBigestContour, moments);

                        CvPoint centerOfTheArm = new CvPoint();

                        if(moments.m00()!= 0) {
                            centerOfTheArm.x(((int) (moments.m10() / moments.m00())));
                            centerOfTheArm.y((int)(moments.m01() / moments.m00()));
                        }

                        cvCircle(original, centerOfTheArm, 4, CvScalar.MAGENTA, 2, 8, 0);

                        if(convexHull != null && !convexHull.isNull() && convexHull.total() > 0) {

                            CvSeq convexityDefects = cvConvexityDefects(approxy, convexHull, memory);

                            if(convexityDefects != null && !convexityDefects.isNull() && convexityDefects.total() > 0) {

                                CvPoint centerOfMass = new CvPoint();

                                int averageDepth = 0;
                                int x = 0;
                                int y = 0;
                                int counter = 0;

                                CvPoint bothPalm = new CvConvexityDefect(cvGetSeqElem(convexityDefects, convexityDefects.total() - 1)).end();

                                CvSeq depthPoints = cvCloneSeq(contour);
                                cvClearSeq(depthPoints);

                                for (int i = 0; i < convexityDefects.total(); i++) {

                                    CvConvexityDefect defect = new CvConvexityDefect(cvGetSeqElem(convexityDefects, i));
                                    cvSeqPush(depthPoints, defect.depth_point());

                                    defect.start(bothPalm);

                                    x += defect.depth_point().x();
                                    y += defect.depth_point().y();
                                    averageDepth += defect.depth();
                                    counter++;

                                    cvCircle(original, defect.depth_point(), 4, CvScalar.YELLOW, 2, 8, 0);
                                    cvCircle(original, defect.start(), 4, CvScalar.BLUE, 2, 8, 0);
                                    cvCircle(original, defect.end(), 4, CvScalar.BLUE, 2, 8, 0);
                                    bothPalm = defect.end();

                                }

                                averageDepth = Math.round(averageDepth / counter);

                                float [] center = new float[2];
                                float [] radius = new float[1];

                                cvMinEnclosingCircle(depthPoints, center, radius);

                                CvPoint centerOfThePalm = cvPoint(((int) ((center[0] + x / counter) / 2)), ((int) ((center[1] + y / counter) / 2)));

                                cvCircle(original, centerOfThePalm, ((int) radius[0]), CvScalar.BLUE, 2, 8, 0);

                                centerOfMass.x(x / counter);
                                centerOfMass.y(y / counter);

                                cvCircle(original, centerOfMass, 4, CvScalar.BLUE, 3, 8, 0);
                            }
                            cvClearSeq(convexityDefects);
                            cvClearSeq(convexHull);
                        }
                    }

                }
                cvShowImage("Hue", h);
                cvShowImage("Saturation", s);
                cvShowImage("Value", v);
                cvShowImage("Mask", mask);
                cvShowImage(originalWindow, original);

                cvReleaseImage(v);
                cvReleaseImage(h);
                cvReleaseImage(s);
                cvReleaseImage(mask);
                if (waitKey(1) == 0) break;
            }
        }

        cvReleaseStructuringElement(kernel);
        cvReleaseStructuringElement(kernel2);
        cvReleaseCapture(camera);
        cvDestroyAllWindows();
    }

    public static int waitKey(int FPS) {
        char key = (char) cvWaitKey(FPS);
        switch (key) {
            case 27:
            case 'q':
                clearAndExit();
                return 0;
            case 'h':
                System.out.println("Hue max: " + (hueMax -= 1));
                break;
            case 'H':
                System.out.println("Hue max: " + (hueMax += 1));
                break;
            case 'x':
                System.out.println("Hue min: " + (hueMin -= 1));
                break;
            case 'X':
                System.out.println("Hue min: " + (hueMin += 1));
                break;
            case 's':
                System.out.println("Saturation max: " + (satMax -= 1));
                break;
            case 'S':
                System.out.println("Saturation max: " + (satMax += 1));
                break;
            case 'c':
                System.out.println("Saturation min: " + (satMin -= 1));
                break;
            case 'C':
                System.out.println("Saturation min: " + (satMin += 1));
                break;
            case 'w':
                System.out.println("Value min: " + (valueMin -= 1));
                break;
            case 'W':
                System.out.println("Value min: " + (valueMin += 1));
                break;
            case 'v':
                System.out.println("Value max: " + (valueMax -= 1));
                break;
            case 'V':
                System.out.println("Value max: " + (valueMax += 1));
                break;
        }
        return 1;
    }

    public static void clearAndExit() {
        cvReleaseImage(original);
        if(mask != null) cvReleaseImage(mask);
        cvReleaseCapture(camera);
    }
}
