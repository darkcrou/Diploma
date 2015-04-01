/**
 * Created by crou on 26.03.15.
 */


import org.bytedeco.javacpp.*;
import org.bytedeco.javacv.CanvasFrame;
import sun.awt.X11GraphicsDevice;

import javax.swing.*;
import java.awt.*;
import java.awt.Point;
import java.awt.event.InputEvent;
import java.util.*;
import java.util.List;
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

    private static Robot robot;

    private static String originalWindow = "original window";
    private static double screenWidth;
    private static double screenHeight;

    private static CvPoint lastTopFingerLocation;
    private static CvPoint lastCursorPosition;
    private static double frameHeight;
    private static double frameWidth;
    private static boolean canClick;
    private static boolean handDetected;

    public static void main(String[] args) {

        camera = cvCreateCameraCapture(0);
        frameHeight = cvGetCaptureProperty(camera, CV_CAP_PROP_FRAME_HEIGHT);
        frameWidth = cvGetCaptureProperty(camera, CV_CAP_PROP_FRAME_WIDTH);
        GraphicsDevice device = CanvasFrame.getDefaultScreenDevice();
        screenHeight = device.getDefaultConfiguration().getBounds().getHeight();
        screenWidth = device.getDefaultConfiguration().getBounds().getWidth();

        try {
            robot = new Robot(CanvasFrame.getDefaultScreenDevice());
        } catch (AWTException e) {
            System.err.print("Cannot initialize ROBOT, so you could not move the cursor");
        }

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

    /**
     * Funkcja wyszukuje nawjekszy zarys na podanej masce.
     * @param mask - maska na ktorej bedzie wyszukiwany zarys
     * @return CvSeq z wyszukanym zarysem*/

     private static CvSeq findTheBiggestContour(IplImage mask) {

        cvClearMemStorage(memory);

        CvSeq contour = new CvContour(null); //Tworzymy nowy ciag dla nowego zarysu

        cvFindContours(mask, memory, contour, Loader.sizeof(CvContour.class), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

        CvSeq theBiggestContour = contour;
         double minContourLength = 400;

        while (contour != null && !contour.isNull()) {
            if (contour.total() > 0) {
                double tempContourArea = cvContourArea(contour);
                double tempContourLength = cvContourPerimeter(contour);
                if (tempContourLength > minContourLength && cvContourArea(theBiggestContour) < tempContourArea) {
                    theBiggestContour = contour;
                }
            }
            contour = contour.h_next();
        }

        return theBiggestContour;
    }

    /**
     * Funkcja ktora rozdziela obrazek na 3 kanaly przestrzeni kolorowej HSV, filtruje kanaly i skleja z powrotem w nowoutworzona maske
     * @param original - oryginalny obrazek dla przetwarzania do maski
     * @return mask - maska */

     private static IplImage createMaskFromHSV(IplImage original) {

        IplImage mask = cvCreateImage(original.cvSize(), IPL_DEPTH_8U, 1);
        hue = cvCreateImage(original.cvSize(), IPL_DEPTH_8U, 1);
        saturation = cvCreateImage(original.cvSize(), IPL_DEPTH_8U, 1);
        value = cvCreateImage(original.cvSize(), IPL_DEPTH_8U, 1);

        cvCvtColor(original, original, CV_BGR2HSV);
        cvSplit(original, hue, saturation, value, null);

        cvInRangeS(hue, cvScalar(hueMin), cvScalar(hueMax), hue);
        cvInRangeS(value, cvScalar(valueMin), cvScalar(valueMax), value);
        cvInRangeS(saturation, cvScalar(satMin), cvScalar(satMax), saturation);

        cvDilate(hue, hue, kernel, 1);
        cvErode(hue, hue, kernel, 2);
//        cvMorphologyEx(hue, hue, null, kernel, CV_MOP_CLOSE, 2);

        cvNot(hue, hue);
        cvAnd(hue, value, mask);
        cvAnd(mask, saturation, mask);
        cvCvtColor(original, original, CV_HSV2BGR);

        return mask;
    }

    private static void drawSequencePolyLines(CvSeq approximation, IplImage on, CvScalar color) {
        CvPoint p0 = new CvPoint(cvGetSeqElem(approximation, approximation.total() - 1));
        for (int i = 0; i < approximation.total(); i++) {
            BytePointer pointer = cvGetSeqElem(approximation, i);
            CvPoint p = new CvPoint(pointer);
            cvLine(on, p0, p, color, 2, 8, 0);
            p0 = p;
        }
    }

    private static void drawSequenceCircles(CvSeq approximation, IplImage on) {
        for (int i = 0; i < approximation.total(); i++) {
            BytePointer pointer = cvGetSeqElem(approximation, i);
            CvPoint p = new CvPoint(pointer);
            cvCircle(on, p, 3, cvScalar(255, 0, 0, 255), 3, 7, 0);
        }
    }

    private static CvSeq findFingers(CvSeq dominantPoints, CvPoint palmCenter, float radius) {
        if(dominantPoints == null) throw new RuntimeException("Given dominantPoints is NULL");
        if(dominantPoints.isNull()) throw new RuntimeException("Given dominantPoints has NULL values");
        int dominantCounts = dominantPoints.total();
        if(dominantCounts == 0) throw new RuntimeException("Given dominantPoints has no values");

        CvSeq fingers = cvCreateSeq(CV_SEQ_ELTYPE_POINT, Loader.sizeof(CvSeq.class), Loader.sizeof(CvPoint.class), memory);

        for(int i = 0; i < dominantCounts; i++) {
            CvPoint dominantPoint = new CvPoint(cvGetSeqElem(dominantPoints, i));
            double distance = vectorLength(vector(dominantPoint, palmCenter));
            double distanceFromBottom = vectorLength(vector(dominantPoint,cvPoint(dominantPoint.x(), (int) frameHeight)));
            if(distance > radius * 1.35 && distance <= radius * 3 && distanceFromBottom > frameHeight * 0.1) {
                cvSeqPush(fingers, dominantPoint);
            }
        }
        if(fingers.total() == 5) handDetected = true;

        return fingers;
    }

    private static CvSeq findDominantPoints(CvSeq contour, int minDist, int angle, CvPoint centerOfPalm) {
        CvSeq points = cvCreateSeq(CV_SEQ_ELTYPE_POINT, Loader.sizeof(contour.getClass()),
                    Loader.sizeof(CvPoint.class), memory);
        int groupCounter = 0;
        double angleCosine = Math.cos(angle);
        LinkedList<CvPoint> group = new LinkedList<>();

        for(int i = 0; i < contour.total(); i++) {
            CvPoint p0 = new CvPoint(cvGetSeqElem(contour, i % contour.total()));

            CvPoint p = new CvPoint(cvGetSeqElem(contour, (i + minDist) % contour.total()));
            CvPoint p1 = new CvPoint(cvGetSeqElem(contour, (i + minDist * 2) % contour.total()));

            int [] vector1 = vector(p0, p);
            int [] vector2 = vector(p1, p);
            double dotProduct = vectorDotProduct(vector1, vector2);
            double cosine = dotProduct / vectorLength(vector1) / vectorLength(vector2);

            if(cosine > angleCosine) {
                group.push(p);
            } else if(group.size() > 0) {
                CvPoint dominant = localDominant(group);
                if(dominant != null) cvSeqPush(points, dominant);
                group.clear();
            }
        }

        return points;
    }

    public static CvPoint localDominant(List segment) {
        if(segment == null) throw new RuntimeException("Your local contour is NULL");
        int contourLength = segment.size();
        CvPoint localDominant = null;
        if(contourLength < 3){
            if(contourLength == 2) {
                CvPoint first = (CvPoint) segment.get(0);
                CvPoint second = (CvPoint) segment.get(1);
                localDominant = new CvPoint();
                localDominant.x((first.x() + second.x())/2);
                localDominant.y((first.y() + second.y())/2);
                return localDominant;
            }
            if(contourLength == 1) return (CvPoint)segment.get(0);
            return localDominant;
        }

        CvPoint first = (CvPoint)segment.get(0);
        CvPoint last = (CvPoint)segment.get(contourLength - 1);
        double theLongestVector = 0;
        int [] mainVector = vector(first, last);

        for(int i = 1; i < contourLength - 1; i++) {
            CvPoint temp = (CvPoint)segment.get(i);
//            int [] tempVector = vector(temp, first);
//            double t = (tempVector[0] * mainVector[0] + tempVector[1] * mainVector[1]) / vectorLength(mainVector);
//            double tempLength = vectorLength(new double [] {(tempVector[0] + mainVector[0] * t), (tempVector[1] + mainVector[1] * t)});

            double tempLength = Math.abs((mainVector[1] * temp.x() - mainVector[0] * temp.y() + last.x()*first.y() - last.y()*first.x())) / Math.sqrt(vectorLength(mainVector));
//            double tempLength = ((last.y() - first.y()) * temp.x() + (first.x() - last.x()) * temp.y() - (first.x() * last.y() - last.x() * first.y()))/
//                    Math.sqrt((last.x() - first.x())*(last.x() - first.x()) + (last.y() - first.y())*(last.y() - first.y()));
            if(tempLength > theLongestVector) {
                localDominant = temp;
                theLongestVector = tempLength;
            }
        }

        return localDominant;
    }

    public static CvPoint localMostPreferred(List segment, CvPoint centerOfPalm) {
        if(segment == null) throw new RuntimeException("Your local contour is NULL");
        int contourLength = segment.size();
        CvPoint localDominant = null;
        if(contourLength < 3){
            if(contourLength == 2) {
                CvPoint first = (CvPoint) segment.get(0);
                CvPoint second = (CvPoint) segment.get(1);
                localDominant = new CvPoint();
                localDominant.x((first.x() + second.x())/2);
                localDominant.y((first.y() + second.y())/2);
                return localDominant;
            }
            if(contourLength == 1) return (CvPoint)segment.get(0);
            return localDominant;
        }

        CvPoint first = (CvPoint)segment.get(0);
        CvPoint last = (CvPoint)segment.get(contourLength - 1);
        CvPoint centered = cvPoint((first.x() + last.x()) / 2, (first.y() + last.y()) / 2);

        double theLongestVector = vectorLength(vector(first, last));
        int [] mainVector = vector(centerOfPalm, centered);

        for(int i = 1; i < contourLength - 1; i++) {
            CvPoint temp = (CvPoint)segment.get(i);
//            int [] tempVector = vector(temp, first);
//            double t = (tempVector[0] * mainVector[0] + tempVector[1] * mainVector[1]) / vectorLength(mainVector);
//            double tempLength = vectorLength(new double [] {(tempVector[0] + mainVector[0] * t), (tempVector[1] + mainVector[1] * t)});

            double tempLength = Math.abs((mainVector[1] * temp.x() - mainVector[0] * temp.y() + last.x()*first.y() - last.y()*first.x())) / Math.sqrt(vectorLength(mainVector));
//            double tempLength = ((last.y() - first.y()) * temp.x() + (first.x() - last.x()) * temp.y() - (first.x() * last.y() - last.x() * first.y()))/
//                    Math.sqrt((last.x() - first.x())*(last.x() - first.x()) + (last.y() - first.y())*(last.y() - first.y()));
            if(tempLength < theLongestVector) {
                localDominant = temp;
                theLongestVector = tempLength;
            }
        }

        return centered;
    }

    private static double vectorLength(int [] vector) {
        return Math.sqrt(vector[0]*vector[0] + vector[1]*vector[1]);
    }

    private static double vectorLength(double [] vector) {
        return Math.sqrt(vector[0]*vector[0] + vector[1]*vector[1]);
    }

    private static int vectorDotProduct(int [] vector1, int [] vector2) {
        return vector1[0]*vector2[0] + vector1[1]*vector2[1];
    }

    private static int[] vector(CvPoint first, CvPoint second) {
        if(first == null) throw new RuntimeException("Cannot count vector 'cuz FIRST point is NULL");
        if(second == null) throw new RuntimeException("Cannot count vector 'cuz SECOND point is NULL");
        if(first.isNull()) throw new RuntimeException("Cannot count vector 'cuz FIRST point has NULL data");
        if(second.isNull()) throw new RuntimeException("Cannot count vector 'cuz SECOND point has NULL data");

        return  new int[]{
                second.x() - first.x(),
                second.y() - first.y()
        };
    }

    public static CvSeq findDeepestPoints(CvSeq convexityDefects) {
        if(convexityDefects == null) throw new RuntimeException("Given convexityDefects is NULL");
        if(convexityDefects.isNull()) throw new RuntimeException("Given convexityDefects has NULL values");
        if(convexityDefects.total() == 0) throw new RuntimeException("Given convexityDefects has no values");

        CvSeq depthPoints = cvCreateSeq(CV_SEQ_ELTYPE_POINT, Loader.sizeof(CvContour.class), Loader.sizeof(CvPoint.class), memory);

        if(convexityDefects != null && !convexityDefects.isNull() && convexityDefects.total() > 0) {

            int averageDepth = 0;
            int x = 0;
            int y = 0;

            CvPoint tempPoint = new CvConvexityDefect(cvGetSeqElem(convexityDefects, convexityDefects.total() - 1)).end();

            for (int i = 0; i < convexityDefects.total(); i++) {
                CvConvexityDefect defect = new CvConvexityDefect(cvGetSeqElem(convexityDefects, i));
                if(defect.depth() > 10) {
                    cvSeqPush(depthPoints, defect.depth_point());

                    defect.start(tempPoint);

                    x += defect.depth_point().x();
                    y += defect.depth_point().y();
                    averageDepth += defect.depth();
                    tempPoint = defect.end();
                }
            }
        }
        return depthPoints;
    }

    public static CvPoint averagePosition(CvSeq contour) {
        if(contour == null) throw new RuntimeException("Given contour is NULL");
        if(contour.isNull()) throw new RuntimeException("Given contour has NULL values");
        if(contour.total() == 0) throw new RuntimeException("Given contour has no values");

        int x = 0;
        int y = 0;
        for (int i = 0; i < contour.total(); i++) {
            CvPoint tmp = new CvPoint(cvGetSeqElem(contour, i));
                x += tmp.x();
                y += tmp.y();
        }

        return cvPoint(x / contour.total(), y / contour.total());
    }

    public static CvPoint theHighestPoint(CvSeq contour) {
        if(contour == null) throw new RuntimeException("Given contour is NULL");
        if(contour.isNull()) throw new RuntimeException("Given contour has NULL values");
        if(contour.total() == 0) throw new RuntimeException("Given contour has no values");

        CvPoint theHighest = new CvPoint(cvGetSeqElem(contour, 0));;
        for(int i = 0; i < contour.total(); i++) {
            CvPoint tmp = new CvPoint(cvGetSeqElem(contour, i));
            if(theHighest.y() > tmp.y()) theHighest = tmp;
        }
        return theHighest;
    }

    private static void process() {
        for (;;) {

            IplImage original = cvQueryFrame(camera);
            cvFlip(original, original, 2);

            IplImage mask = createMaskFromHSV(original);

            CvSeq theBiggestContour = findTheBiggestContour(mask);

            if (theBiggestContour != null && !theBiggestContour.isNull()) {

                CvMoments moments = new CvMoments();

                cvMoments(theBiggestContour, moments);

                CvPoint centerOfTheArm = new CvPoint();

                if(moments.m00()!= 0) {
                    centerOfTheArm.x(((int) (moments.m10() / moments.m00())));
                    centerOfTheArm.y((int) (moments.m01() / moments.m00()));
                }
                cvCircle(original, centerOfTheArm, 4, CvScalar.MAGENTA, 2, 8, 0);

                CvSeq approximation = cvApproxPoly(theBiggestContour, Loader.sizeof(CvContour.class), memory, CV_POLY_APPROX_DP, cvContourPerimeter(theBiggestContour) * 0.0015, 1);
                CvSeq convexHull = cvConvexHull2(approximation, memory, CV_CLOCKWISE, 0);
                CvSeq convexHullForDrawing = cvConvexHull2(approximation, memory, CV_CLOCKWISE, 1);
                drawSequencePolyLines(convexHullForDrawing, original, cvScalar(0xff0000));
                CvSeq convexityDefects = cvConvexityDefects(approximation, convexHull, memory);
                CvSeq theDeepestPoints;

                if(!convexityDefects.isNull() && convexityDefects.total() > 1) {
                    theDeepestPoints = findDeepestPoints(convexityDefects);

                    float [] tempEnclosingCircleCenter = new float[2];
                    float [] enclosingCircleRadius = new float[1];

                    if(theDeepestPoints.total() > 1) {
                        cvMinEnclosingCircle(theDeepestPoints, tempEnclosingCircleCenter, enclosingCircleRadius);
                        CvPoint enclosingCircleCenter = cvPoint(((int) tempEnclosingCircleCenter[0]), ((int)tempEnclosingCircleCenter[1]));
                        CvPoint averageDepthPointPosition = averagePosition(theDeepestPoints);

                        enclosingCircleRadius[0] = (float) (enclosingCircleRadius[0] - vectorLength(vector(enclosingCircleCenter, averageDepthPointPosition)) / 2);

                        CvPoint averageCenter = cvPoint((averageDepthPointPosition.x() + enclosingCircleCenter.x()) / 2,
                                (averageDepthPointPosition.y() + enclosingCircleCenter.y()) / 2);

                        int dominantDistance = (theBiggestContour.total() * 0.03) < 25? 25: (int) (theBiggestContour.total() * 0.03);
                        CvSeq dominantPoints = findDominantPoints(theBiggestContour, dominantDistance, 70, averageCenter);
                        drawSequenceCircles(dominantPoints, original);

                        if(dominantPoints.total() > 0) {
                            CvSeq fingers = findFingers(dominantPoints, averageCenter, enclosingCircleRadius[0]);

                            if(fingers.total() == 1) {
                                CvPoint currentFingerLocation = theHighestPoint(fingers);
                                if(lastTopFingerLocation != null && handDetected) {
                                    int newCurPosX = (int) ((currentFingerLocation.x() / frameWidth) * screenWidth);
                                    int newCurPosY = (int) ((currentFingerLocation.y() / frameHeight) * screenHeight);
                                    if(lastCursorPosition == null) lastCursorPosition = cvPoint(newCurPosX, newCurPosY);
                                    if(Math.abs(newCurPosX - lastCursorPosition.x()) / (lastCursorPosition.x()) < 0.01 &&
                                            Math.abs(newCurPosY - lastCursorPosition.y()) / (lastCursorPosition.y()) < 0.01 ) {
                                        robot.mouseMove(newCurPosX, newCurPosY);
                                    }
                                    lastCursorPosition.x(newCurPosX);
                                    lastCursorPosition.y(newCurPosY);
                                    canClick = true;
                                }
                                lastTopFingerLocation = currentFingerLocation;
                            } else if(fingers.total() == 2 && canClick && handDetected) {
                                robot.mousePress(InputEvent.BUTTON1_MASK);
                                robot.delay(100);
                                robot.mouseRelease(InputEvent.BUTTON1_MASK);
                                canClick = false;
                            }

                            System.out.println("Fingers found: " + fingers.total());
                            cvCircle(original, averageCenter, (int) Math.abs(enclosingCircleRadius[0]), CvScalar.RED, 2, 8, 0);

                            drawSequencePolyLines(fingers, original, cvScalar(0, 255, 0, 255));
                            cvClearSeq(fingers);
                        }
                    }

                    cvClearSeq(theDeepestPoints);
                }
                cvClearSeq(convexityDefects);
                cvClearSeq(convexHull);
                cvClearSeq(approximation);
                cvClearSeq(theBiggestContour);
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
//            cvReleaseImage(original);

            if (waitKey(1) == 0) break;
        }

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
