/**
 * Created by crou on 26.03.15.
 */


import org.bytedeco.javacpp.*;

import javax.swing.*;
import java.awt.*;
import java.util.HashSet;
import java.util.concurrent.ConcurrentHashMap;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_highgui.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

public class Main {

    private static ConcurrentHashMap<String, Integer> settingsHolder = new ConcurrentHashMap<String, Integer>();

    private static boolean byHSV;

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

    private static boolean hueInverted;
    private static boolean saturationInverted;
    private static boolean valueInverted;
    private static boolean channelYInverted;
    private static boolean chanelCrInverted;
    private static boolean chanelCbInverted;

    private static CvCapture camera;

    private static IplImage original;

    private static IplImage hue;
    private static IplImage saturation;
    private static IplImage value;
    private static IplImage chanelY;
    private static IplImage chanelCr;
    private static IplImage chanelCb;

    private static byte[] calibrationHueAndValue;

    private static IplConvKernel kernel = cvCreateStructuringElementEx(3, 3, 1, 1, CV_SHAPE_ELLIPSE);
    private static IplConvKernel kernel2 = cvCreateStructuringElementEx(7, 7, 4, 4, CV_SHAPE_ELLIPSE);
    private static CvMemStorage memory = cvCreateMemStorage(0xffff);
    private static CvSeq approximation;
    private static CvPoint centerOfPalm;
    private static HashSet<CvPoint> fingerTips = new HashSet<>();

    private static String originalWindow = "original window";

    public static void main(String[] args) {

        camera = cvCreateCameraCapture(0);

        if (camera != null) {
            cvNamedWindow(originalWindow, 0);
            cvNamedWindow("Value", 0);
            cvNamedWindow("Hue", 0);
            cvNamedWindow("Saturation", 0);

            process();

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

    private static CvSeq findTheBiggestContour(IplImage mask) {

        cvClearMemStorage(memory);

        CvSeq contour = new CvContour(null);

        cvFindContours(mask, memory, contour, Loader.sizeof(CvContour.class), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

        CvSeq theBiggestContour = contour;

        while (contour != null && !contour.isNull()) {
            if (contour.total() > 0) {
                if (cvContourArea(theBiggestContour) < cvContourArea(contour)) {
                    theBiggestContour = contour;
                }
            }
            contour = contour.h_next();
        }

        return theBiggestContour;
    }

    private static IplImage createMaskFromHSV(IplImage original) {

        IplImage mask = cvCreateImage(original.cvSize(), IPL_DEPTH_8U, 1);
        hue = cvCreateImage(original.cvSize(), IPL_DEPTH_8U, 1);
        saturation = cvCreateImage(original.cvSize(), IPL_DEPTH_8U, 1);
        value = cvCreateImage(original.cvSize(), IPL_DEPTH_8U, 1);

        cvCvtColor(original, original, CV_BGR2HSV);
        cvSplit(original, hue, saturation, value, null);

        cvInRangeS(hue, cvScalar(hueMin), cvScalar(hueMax), hue);
        cvInRangeS(saturation, cvScalar(valueMin), cvScalar(valueMax), saturation);
        cvInRangeS(value, cvScalar(satMin), cvScalar(satMax), value);

        cvErode(hue, hue, kernel, 1);
        cvDilate(hue, hue, kernel, 1);

        cvNot(hue, hue);
        cvAnd(hue, value, mask);
        cvAnd(mask, saturation, mask);
        cvCvtColor(original, original, CV_HSV2BGR);

        return mask;
    }

    private static void drawSequence(CvSeq approximation, IplImage on) {
        CvPoint p0 = new CvPoint(cvGetSeqElem(approximation, approximation.total() - 1));
        for (int i = 0; i < approximation.total(); i++) {
            BytePointer pointer = cvGetSeqElem(approximation, i);
            CvPoint p = new CvPoint(pointer);
            cvLine(on, p0, p, CvScalar.GREEN, 2, 8, 0);
            p0 = p;
        }
    }

    private static CvSeq findDominantPoints(CvSeq contour, int dist, int angle) {

        CvSeq points = cvCreateSeq(CV_SEQ_ELTYPE_POINT, Loader.sizeof(CvContour.class), Loader.sizeof(CvPoint.class), memory);

        for(int i = 0; i < contour.total() - dist * 2; i++) {
            CvPoint p0 = new CvPoint(cvGetSeqElem(contour, i));
            CvPoint p = new CvPoint(cvGetSeqElem(contour, (i + dist)));
            CvPoint p1 = new CvPoint(cvGetSeqElem(contour, (i + dist * 2)));

            int [] vector1 = new int[]{
                    p.x() - p0.x(),
                    p.y() - p0.y()
            };
            int [] vector2 = new int[]{
                    p.x() - p1.x(),
                    p.y() - p1.y()
            };

            int dotProduct = vector1[0]*vector2[0] + vector1[1]*vector2[1];
            int cosine = (int) (dotProduct / (Math.sqrt(vector1[0]*vector1[0] + vector1[1]*vector1[1]) * Math.sqrt(vector2[0]*vector2[0] + vector2[1]*vector2[1])));

            if(cosine < Math.cos(angle)) {
                cvSeqPush(points, p);
            }

        }

        return points;
    }

    private static void process() {
        for (;;) {

            original = cvQueryFrame(camera);

            IplImage mask = createMaskFromHSV(original);

            CvSeq theBiggestContour = findTheBiggestContour(mask);

            if (theBiggestContour != null && !theBiggestContour.isNull()) {

                approximation = cvApproxPoly(theBiggestContour, Loader.sizeof(CvContour.class), memory, CV_POLY_APPROX_DP, cvContourPerimeter(theBiggestContour) * 0.0015, 1);

                if(approximation != null && !approximation.isNull()) {

                    System.out.println("Approxi length: " + approximation.total());

                    CvSeq dominantPoints = findDominantPoints(approximation, ((int) (approximation.total() * 0.2)), 120);
                    System.out.println("Dominant points found: " + dominantPoints.total());
                    drawSequence(approximation, original);

                    CvMoments moments = new CvMoments();

                    cvMoments(theBiggestContour, moments);

                    CvPoint centerOfTheArm = new CvPoint();

                    if(moments.m00()!= 0) {
                        centerOfTheArm.x(((int) (moments.m10() / moments.m00())));
                        centerOfTheArm.y((int)(moments.m01() / moments.m00()));
                    }

                    cvCircle(original, centerOfTheArm, 4, CvScalar.MAGENTA, 2, 8, 0);

                    CvSeq convexHull = cvConvexHull2(approximation, memory, CV_CLOCKWISE, 0);

                    if(convexHull != null && !convexHull.isNull() && convexHull.total() > 0) {

                        CvSeq convexityDefects = cvConvexityDefects(approximation, convexHull, memory);

                        if(convexityDefects != null && !convexityDefects.isNull() && convexityDefects.total() > 0) {

                            CvPoint centerOfMass = new CvPoint();

                            int averageDepth = 0;
                            int x = 0;
                            int y = 0;

                            CvPoint tempPoint = new CvConvexityDefect(cvGetSeqElem(convexityDefects, convexityDefects.total() - 1)).end();

                            CvSeq depthPoints = cvCreateSeq(CV_SEQ_ELTYPE_POINT, Loader.sizeof(CvContour.class), Loader.sizeof(CvPoint.class), memory);

                            for (int i = 0; i < convexityDefects.total(); i++) {

                                CvConvexityDefect defect = new CvConvexityDefect(cvGetSeqElem(convexityDefects, i));
                                if(defect.depth() > 10) {
                                    cvSeqPush(depthPoints, defect.depth_point());

                                    defect.start(tempPoint);

                                    x += defect.depth_point().x();
                                    y += defect.depth_point().y();
                                    averageDepth += defect.depth();

                                    cvCircle(original, defect.depth_point(), 4, CvScalar.YELLOW, 2, 8, 0);
                                    cvCircle(original, defect.start(), 4, CvScalar.BLUE, 2, 8, 0);
                                    cvCircle(original, defect.end(), 4, CvScalar.BLUE, 2, 8, 0);
                                    tempPoint = defect.end();
                                }
                            }

                            if(depthPoints.total() > 0) {
                                averageDepth = Math.round(averageDepth / depthPoints.total());

                                float [] center = new float[2];
                                float [] radius = new float[1];

                                cvMinEnclosingCircle(depthPoints, center, radius);

                                CvPoint centerOfThePalm = cvPoint(((int) ((center[0] + x / depthPoints.total()) / 2)), ((int) ((center[1] + y / depthPoints.total()) / 2)));

                                cvCircle(original, centerOfThePalm, ((int) (radius[0] + averageDepth) / 2), CvScalar.BLUE, 2, 8, 0);

                                centerOfMass.x(x / depthPoints.total());
                                centerOfMass.y(y / depthPoints.total());

                                cvCircle(original, centerOfMass, 4, CvScalar.BLUE, 3, 8, 0);
                            }

                        }
                        cvClearSeq(convexityDefects);
                        cvClearSeq(convexHull);
                    }
                }

            }
            cvShowImage("Hue", hue);
            cvShowImage("Saturation", saturation);
            cvShowImage("Value", value);
//            cvShowImage("Mask", mask);
            cvShowImage(originalWindow, original);

            cvReleaseImage(value);
            cvReleaseImage(hue);
            cvReleaseImage(saturation);
            cvReleaseImage(mask);
            if (waitKey(1) == 0) break;
        }

        Thread images = new Thread();

    }

    private static void configurationWindow() {
        Runnable configWindow = () -> {
            JFrame configurationFrame = new JFrame("Configuration frame");
            configurationFrame.setMinimumSize(new Dimension(300, 100));

            JSlider hueMinSlider = new JSlider(0, 180);
            JSlider hueMaxSlider = new JSlider(0, 180);
            hueMinSlider.addPropertyChangeListener((e) -> {
                JSlider slider = (JSlider) e.getSource();
                if(!slider.getValueIsAdjusting()){
                    settingsHolder.put("hueMin", slider.getValue());
                }
            });

            JSlider saturationMinSlider = new JSlider(0, 255);
            JSlider saturationMaxSlider = new JSlider(0, 255);
            JSlider valueMinSlider = new JSlider(0, 255);
            JSlider valueMaxSlider = new JSlider(0, 255);
            JSlider chanelYMinSlider = new JSlider(0, 255);
            JSlider chanelYMaxSlider = new JSlider(0, 255);
            JSlider chanelCrMinSlider = new JSlider(0, 255);
            JSlider chanelCrMaxSlider = new JSlider(0, 255);
            JSlider chanelCbMinSlider = new JSlider(0, 255);
            JSlider chanelCbMaxSlider = new JSlider(0, 255);

            JCheckBox colorSchema = new JCheckBox("Color schema selector");

            configurationFrame.add(hueMinSlider);
            configurationFrame.setVisible(true);
            configurationFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);

        };

        Thread windowThread = new Thread(configWindow);
        windowThread.start();
    }
}
