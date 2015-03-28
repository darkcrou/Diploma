/**
 * Created by crou on 26.03.15.
 */


import javafx.util.converter.PercentageStringConverter;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;

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

    private static CvCapture camera;
    private static IplImage original;
    private static IplImage mask;
    private static IplImage h;
    private static IplImage v;

    private static byte[] calibrationHueAndValue;

    public static void main(String[] args) {

        String originalWindow = "original window";
        String maskWindow = "moded window";
        String valueWindow = "value window";

        camera = cvCreateCameraCapture(0);

        IplConvKernel kernel = cvCreateStructuringElementEx(3, 3, 2, 2, CV_SHAPE_ELLIPSE);
        IplConvKernel kernel2 = cvCreateStructuringElementEx(7, 7, 4, 4, CV_SHAPE_ELLIPSE);

        if (camera != null) {
            cvNamedWindow(originalWindow, 0);
            cvNamedWindow(maskWindow, 0);

            CvMemStorage memory = cvCreateMemStorage(0xffff);

            for (;;) {

                CvSeq contour = new CvContour();

                original = cvQueryFrame(camera);

                if (calibrated) {
                    mask = cvCreateImage(original.cvSize(), IPL_DEPTH_8U, 1);

                    cvCvtColor(original, original, CV_BGR2HSV);
                    cvInRangeS(original, cvScalar(hueMin, satMin, valueMin, 0), cvScalar(hueMax, satMax, valueMax, 0), mask);

                    cvCvtColor(original, original, CV_HSV2BGR);

                    cvNot(mask, mask);

                    cvErode(mask, mask, kernel, 2);

                    IplImage originalMask = cvCloneImage(mask);

                    int count = cvFindContours(mask, memory, contour, Loader.sizeof(CvContour.class), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

                    if (contour != null && !contour.isNull() && count > 0) {

                        CvSeq theBigestContour = contour;

                        while (contour != null && !contour.isNull()) {
                            if (contour.elem_size() > 0) {
                                CvSize2D32f size = cvMinAreaRect2(theBigestContour, memory).size();
                                CvSize2D32f curSize = cvMinAreaRect2(contour, memory).size();

                                if (size.height() * size.width() < curSize.height() * curSize.width())
                                    theBigestContour = contour;
                            }
                            contour = contour.h_next();
                        }

                        CvSeq approxy = cvApproxPoly(theBigestContour, Loader.sizeof(CvContour.class), memory, CV_POLY_APPROX_DP, 10, 1);

                        if(approxy != null && !approxy.isNull()) {
                            if (approxy != null && !approxy.isNull()) {
                                CvPoint p0 = new CvPoint(cvGetSeqElem(approxy, approxy.total() - 1));
                                for (int i = 0; i < approxy.total(); i++) {
                                    BytePointer pointer = cvGetSeqElem(approxy, i);
                                    CvPoint p = new CvPoint(pointer);
                                    cvLine(original, p0, p, CvScalar.GREEN, 2, 8, 0);
                                    p0 = p;
                                }
                            }

                            CvSeq convexHull = cvConvexHull2(approxy, memory, CV_COUNTER_CLOCKWISE, 0);

                            CvMoments moments = new CvMoments();

                            cvMoments(theBigestContour, moments);

                            CvPoint centerOfMass = new CvPoint();

                            if(moments.m00() != 0) {
                                centerOfMass.x((int) (moments.m10() / moments.m00()));
                                centerOfMass.y(((int) (moments.m01() / moments.m00())));
                            }

                            cvCircle(original, centerOfMass, 3, CvScalar.GREEN, 1,1,1);

                            if(convexHull != null && !convexHull.isNull() && convexHull.total() > 0) {
//                                CvPoint hp0 = new CvPoint(cvGetSeqElem(convexHull, convexHull.total() - 1));
//                                for (int i = 0; i < convexHull.total(); i++) {
//                                    CvPoint hp = new CvPoint(cvGetSeqElem(convexHull, i));
//                                    cvLine(original, hp0, hp, CvScalar.RED, 2, 8, 0);
//                                    hp0 = hp;
//                                }

                                CvSeq convexityDefects = cvConvexityDefects(approxy, convexHull, memory);

                                if(convexityDefects != null && !convexityDefects.isNull() && convexityDefects.total() > 0) {
                                    for (int i = 0; i < convexityDefects.total(); i++) {
                                        CvConvexityDefect defect = new CvConvexityDefect(cvGetSeqElem(convexityDefects, i));
                                        cvCircle(original, defect.depth_point(), 4, CvScalar.YELLOW);
                                        cvCircle(original, defect.start(), 4, CvScalar.BLUE);
                                        cvCircle(original, defect.end(), 4, CvScalar.BLUE);
                                    }
                                }
                                cvClearSeq(convexityDefects);
                                cvClearSeq(convexHull);
                            }
                        }
                    }

                    cvShowImage(maskWindow, originalMask);

                    cvReleaseImage(mask);
                    cvReleaseImage(originalMask);

                }
                cvShowImage(originalWindow, original);

                if (waitKey(25) == 0) break;

            }
        }

        cvReleaseStructuringElement(kernel);
        cvReleaseCapture(camera);
        cvDestroyAllWindows();

    }

    public static byte[] calibration(IplImage calibrateImage) {

        CvSize size = cvSize(20, 20);

        CvPoint center = cvPoint(calibrateImage.width() / 2, calibrateImage.height() / 2);

        byte maxHue = 0;
        byte minHue = 0;
        byte maxValue = 0;
        byte minValue = 0;

        for (int x = center.x() - size.width() / 2; x < center.x() + size.width() / 2; x++) {
            for (int y = center.y() - size.height() / 2; y < center.y() + size.height() / 2; y++) {
                byte hue = calibrateImage.imageData().get(y * calibrateImage.widthStep() + x * 3);
                byte value = calibrateImage.imageData().get(y * calibrateImage.widthStep() + x * 3 + 2);

                if (maxHue < hue && hue != maxHue) maxHue = hue;
                else minHue = hue;
                if (maxValue < value && value != maxValue) maxValue = value;
                else minValue = value;
            }
        }

        System.out.println("minHue" + minHue);
        System.out.println("maxHue" + maxHue);
        System.out.println("minValue" + minValue);
        System.out.println("maxValue" + maxValue);

        calibrated = true;

        return new byte[]{minHue, maxHue, minValue, maxValue};
    }

    public static int waitKey(int FPS) {
        char key = (char) cvWaitKey(FPS);
        switch (key) {
            case 27:
            case 'q':
                clearAndExit();
                return 0;
            case 'p':
                calibrationHueAndValue = calibration(original);
                System.out.println("Calibrated.");
                break;
            case 'h':
                System.out.println("Hue max: " + (hueMax -= 5));
                break;
            case 'H':
                System.out.println("Hue max: " + (hueMax += 5));
                break;
            case 'x':
                System.out.println("Hue min: " + (hueMin -= 5));
                break;
            case 'X':
                System.out.println("Hue min: " + (hueMin += 5));
                break;
            case 's':
                System.out.println("Saturation max: " + (satMax -= 5));
                break;
            case 'S':
                System.out.println("Saturation max: " + (satMax += 5));
                break;
            case 'c':
                System.out.println("Saturation min: " + (satMin -= 5));
                break;
            case 'C':
                System.out.println("Saturation min: " + (satMin += 5));
                break;
            case 'w':
                System.out.println("Value min: " + (valueMin -= 5));
                break;
            case 'W':
                System.out.println("Value min: " + (valueMin += 5));
                break;
            case 'v':
                System.out.println("Value max: " + (valueMax -= 5));
                break;
            case 'V':
                System.out.println("Value max: " + (valueMax += 5));
                break;
        }
        return 1;
    }

    public static void clearAndExit() {
        cvReleaseImage(original);
        cvReleaseImage(mask);
        cvReleaseCapture(camera);
    }
}
