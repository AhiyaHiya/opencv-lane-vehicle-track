

#include <math.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <vector>

#include "utils.h"

//#define USE_VIDEO 1

#if defined(_WIN32)
#undef MIN
#undef MAX
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define MIN(a, b) ((a) > (b) ? (b) : (a))
#endif

void crop(IplImage* src, IplImage* dest, CvRect rect)
{
    cvSetImageROI(src, rect);
    cvCopy(src, dest);
    cvResetImageROI(src);
}

struct Lane
{
    Lane() {}
    Lane(CvPoint a, CvPoint b, float angle, float kl, float bl)
        : p0(a)
        , p1(b)
        , angle(angle)
        , votes(0)
        , visited(false)
        , found(false)
        , k(kl)
        , b(bl)
    {
    }

    CvPoint p0, p1;
    int     votes;
    bool    visited, found;
    float   angle, k, b;
};

struct Status
{
    Status()
        : reset(true)
        , lost(0)
    {
    }
    ExpMovingAverage k, b;
    bool             reset;
    int              lost;
};

struct Vehicle
{
    CvPoint      bmin, bmax;
    int          symmetryX;
    bool         valid;
    unsigned int lastUpdate;
};

struct VehicleSample
{
    CvPoint      center;
    float        radi;
    unsigned int frameDetected;
    int          vehicleIndex;
};

#define GREEN CV_RGB(0, 255, 0)
#define RED CV_RGB(255, 0, 0)
#define BLUE CV_RGB(255, 0, 255)
#define PURPLE CV_RGB(255, 0, 255)

auto laneR    = Status{};
auto laneL    = Status{};
auto vehicles = std::vector< Vehicle >{};
auto samples  = std::vector< VehicleSample >{};

enum
{
    SCAN_STEP           = 5,   // in pixels
    LINE_REJECT_DEGREES = 10,  // in degrees
    BW_TRESHOLD         = 250, // edge response strength to recognize for 'WHITE'
    BORDERX             = 10,  // px, skip this much from left & right borders
    MAX_RESPONSE_DIST   = 5,   // px

    CANNY_MIN_TRESHOLD = 1,   // edge detector minimum hysteresis threshold
    CANNY_MAX_TRESHOLD = 100, // edge detector maximum hysteresis threshold

    HOUGH_TRESHOLD        = 50,  // line approval vote threshold
    HOUGH_MIN_LINE_LENGTH = 50,  // remove lines shorter than this treshold
    HOUGH_MAX_LINE_GAP    = 100, // join lines to one with smaller than this gaps

    CAR_DETECT_LINES  = 4,  // minimum lines for a region to pass validation as a 'CAR'
    CAR_H_LINE_LENGTH = 10, // minimum horizontal line length from car body in px

    MAX_VEHICLE_SAMPLES = 30, // max vehicle detection sampling history
    CAR_DETECT_POSITIVE_SAMPLES =
        MAX_VEHICLE_SAMPLES - 2,    // probability positive matches for valid car
    MAX_VEHICLE_NO_UPDATE_FREQ = 15 // remove car after this much no update frames
};

#define K_VARY_FACTOR 0.2f
#define B_VARY_FACTOR 20
#define MAX_LOST_FRAMES 30

void FindResponses(IplImage* img, int startX, int endX, int y, std::vector< int >& list)
{
    // scans for single response: /^\_

    const auto row = y * img->width * img->nChannels;
    auto       ptr = reinterpret_cast< unsigned char* >(img->imageData);

    auto step  = (endX < startX) ? -1 : 1;
    auto range = (endX > startX) ? endX - startX + 1 : startX - endX + 1;

    for (auto x = startX; range > 0; x += step, range--)
    {
        if (ptr[row + x] <= BW_TRESHOLD)
            continue; // skip black: loop until white pixels show up

        // first response found
        int idx = x + step;

        // skip same response(white) pixels
        while (range > 0 && ptr[row + idx] > BW_TRESHOLD)
        {
            idx += step;
            range--;
        }

        // reached black again
        if (ptr[row + idx] <= BW_TRESHOLD)
        {
            list.push_back(x);
        }

        x = idx; // begin from new pos
    }
}

unsigned char pixel(IplImage* img, int x, int y)
{
    return (unsigned char)img->imageData[(y * img->width + x) * img->nChannels];
}

int findSymmetryAxisX(IplImage* half_frame, CvPoint bmin, CvPoint bmax)
{

    float value = 0;
    int   axisX = -1; // not found

    auto xmin       = bmin.x;
    auto ymin       = bmin.y;
    auto xmax       = bmax.x;
    auto ymax       = bmax.y;
    auto half_width = half_frame->width / 2;
    auto maxi       = 1;

    for (int x = xmin, j = 0; x < xmax; x++, j++)
    {
        float HS = 0;
        for (auto y = ymin; y < ymax; y++)
        {
            auto row = y * half_frame->width * half_frame->nChannels;
            for (auto step = 1; step < half_width; step++)
            {
                auto          neg = x - step;
                auto          pos = x + step;
                unsigned char Gneg =
                    (neg < xmin)
                        ? 0
                        : (unsigned char)half_frame->imageData[row + neg * half_frame->nChannels];
                unsigned char Gpos =
                    (pos >= xmax)
                        ? 0
                        : (unsigned char)half_frame->imageData[row + pos * half_frame->nChannels];
                HS += abs(Gneg - Gpos);
            }
        }

        if (axisX == -1 || value > HS)
        { // find minimum
            axisX = x;
            value = HS;
        }
    }

    return axisX;
}

bool hasVertResponse(IplImage* edges, int x, int y, int ymin, int ymax)
{
    bool has = (pixel(edges, x, y) > BW_TRESHOLD);
    if (y - 1 >= ymin)
        has &= (pixel(edges, x, y - 1) < BW_TRESHOLD);
    if (y + 1 < ymax)
        has &= (pixel(edges, x, y + 1) < BW_TRESHOLD);
    return has;
}

int horizLine(IplImage* edges, int x, int y, CvPoint bmin, CvPoint bmax, int maxHorzGap)
{

    // scan to right
    auto right = 0;
    auto gap   = maxHorzGap;
    for (auto xx = x; xx < bmax.x; xx++)
    {
        if (hasVertResponse(edges, xx, y, bmin.y, bmax.y))
        {
            right++;
            gap = maxHorzGap; // reset
        }
        else
        {
            gap--;
            if (gap <= 0)
            {
                break;
            }
        }
    }

    auto left = 0;
    gap       = maxHorzGap;
    for (auto xx = x - 1; xx >= bmin.x; xx--)
    {
        if (hasVertResponse(edges, xx, y, bmin.y, bmax.y))
        {
            left++;
            gap = maxHorzGap; // reset
        }
        else
        {
            gap--;
            if (gap <= 0)
            {
                break;
            }
        }
    }

    return left + right;
}

bool vehicleValid(IplImage* half_frame, IplImage* edges, Vehicle* v, int& index)
{

    index = -1;

    // first step: find horizontal symmetry axis
    v->symmetryX = findSymmetryAxisX(half_frame, v->bmin, v->bmax);
    if (v->symmetryX == -1)
        return false;

    // second step: cars tend to have a lot of horizontal lines
    auto hlines = 0;
    for (auto y = v->bmin.y; y < v->bmax.y; y++)
    {
        if (horizLine(edges, v->symmetryX, y, v->bmin, v->bmax, 2) > CAR_H_LINE_LENGTH)
        {
#if _DEBUG
            cvCircle(half_frame, cvPoint(v->symmetryX, y), 2, PURPLE);
#endif
            hlines++;
        }
    }

    auto midy = (v->bmax.y + v->bmin.y) / 2;

    // third step: check with previous detected samples if car already exists
    auto  numClose    = 0;
    float closestDist = 0;
    for (auto i = 0; i < samples.size(); i++)
    {
        auto  dx   = samples[i].center.x - v->symmetryX;
        auto  dy   = samples[i].center.y - midy;
        float Rsqr = dx * dx + dy * dy;

        if (Rsqr <= samples[i].radi * samples[i].radi)
        {
            numClose++;
            if (index == -1 || Rsqr < closestDist)
            {
                index       = samples[i].vehicleIndex;
                closestDist = Rsqr;
            }
        }
    }

    return (hlines >= CAR_DETECT_LINES || numClose >= CAR_DETECT_POSITIVE_SAMPLES);
}

void removeOldVehicleSamples(unsigned int currentFrame)
{
    // statistical sampling - clear very old samples
    std::vector< VehicleSample > sampl;
    for (auto i = 0; i < samples.size(); i++)
    {
        if (currentFrame - samples[i].frameDetected < MAX_VEHICLE_SAMPLES)
        {
            sampl.push_back(samples[i]);
        }
    }
    samples = sampl;
}

void removeSamplesByIndex(int index)
{
    // statistical sampling - clear very old samples
    std::vector< VehicleSample > sampl;
    for (auto i = 0; i < samples.size(); i++)
    {
        if (samples[i].vehicleIndex != index)
        {
            sampl.push_back(samples[i]);
        }
    }
    samples = sampl;
}

void removeLostVehicles(unsigned int currentFrame)
{
    // remove old unknown/false vehicles & their samples, if any
    for (auto i = 0; i < vehicles.size(); i++)
    {
        if (vehicles[i].valid &&
            currentFrame - vehicles[i].lastUpdate >= MAX_VEHICLE_NO_UPDATE_FREQ)
        {
            printf("\tremoving inactive car, index = %d\n", i);
            removeSamplesByIndex(i);
            vehicles[i].valid = false;
        }
    }
}

void vehicleDetection(IplImage*                half_frame,
                      CvHaarClassifierCascade* cascade,
                      CvMemStorage*            haarStorage)
{

    static auto frame = 0;
    frame++;
    printf("*** vehicle detector frame: %d ***\n", frame);

    removeOldVehicleSamples(frame);

    // Haar Car detection
    const double scale_factor   = 1.05; // every iteration increases scan window by 5%
    const auto   min_neighbours = 2; // minus 1, number of rectangles, that the object consists of
    CvSeq*       rects          = cvHaarDetectObjects(
        half_frame, cascade, haarStorage, scale_factor, min_neighbours, CV_HAAR_DO_CANNY_PRUNING);

    // Canny edge detection of the minimized frame
    if (rects->total > 0)
    {
        printf("\thaar detected %d car hypotheses\n", rects->total);
        auto edges = cvCreateImage(cvSize(half_frame->width, half_frame->height), IPL_DEPTH_8U, 1);
        cvCanny(half_frame, edges, CANNY_MIN_TRESHOLD, CANNY_MAX_TRESHOLD);

        /* validate vehicles */
        for (auto i = 0; i < rects->total; i++)
        {
            auto rc = reinterpret_cast< CvRect* >(cvGetSeqElem(rects, i));

            auto v  = Vehicle{};
            v.bmin  = cvPoint(rc->x, rc->y);
            v.bmax  = cvPoint(rc->x + rc->width, rc->y + rc->height);
            v.valid = true;

            auto index = 0;
            if (vehicleValid(half_frame, edges, &v, index))
            { // put a sample on that position

                if (index == -1)
                { // new car detected

                    v.lastUpdate = frame;

                    // re-use already created but inactive vehicles
                    for (auto j = 0; j < vehicles.size(); j++)
                    {
                        if (vehicles[j].valid == false)
                        {
                            index = j;
                            break;
                        }
                    }
                    if (index == -1)
                    { // all space used
                        index = vehicles.size();
                        vehicles.push_back(v);
                    }
                    printf("\tnew car detected, index = %d\n", index);
                }
                else
                {
                    // update the position from new data
                    vehicles[index]            = v;
                    vehicles[index].lastUpdate = frame;
                    printf("\tcar updated, index = %d\n", index);
                }

                VehicleSample vs;
                vs.frameDetected = frame;
                vs.vehicleIndex  = index;
                vs.radi          = (MAX(rc->width, rc->height)) /
                          4; // radius twice smaller - prevent false positives
                vs.center = cvPoint((v.bmin.x + v.bmax.x) / 2, (v.bmin.y + v.bmax.y) / 2);
                samples.push_back(vs);
            }
        }

        cvShowImage("Half-frame[edges]", edges);
        cvMoveWindow("Half-frame[edges]", half_frame->width * 2 + 10, half_frame->height);
        cvReleaseImage(&edges);
    }
    else
    {
        printf("\tno vehicles detected in current frame!\n");
    }

    removeLostVehicles(frame);

    printf("\ttotal vehicles on screen: %d\n", vehicles.size());
}

void drawVehicles(IplImage* half_frame)
{

    // show vehicles
    for (auto i = 0; i < vehicles.size(); i++)
    {
        Vehicle* v = &vehicles[i];
        if (v->valid)
        {
            cvRectangle(half_frame, v->bmin, v->bmax, GREEN, 1);

            auto midY = (v->bmin.y + v->bmax.y) / 2;
            cvLine(half_frame,
                   cvPoint(v->symmetryX, midY - 10),
                   cvPoint(v->symmetryX, midY + 10),
                   PURPLE);
        }
    }

    // show vehicle position sampling
    /*for (int i = 0; i < samples.size(); i++) {
            cvCircle(half_frame, cvPoint(samples[i].center.x, samples[i].center.y), samples[i].radi,
    RED);
    }*/
}

void processSide(std::vector< Lane > lanes, IplImage* edges, bool right)
{

    Status* side = right ? &laneR : &laneL;

    // response search
    auto           w      = edges->width;
    auto           h      = edges->height;
    const auto     BEGINY = 0;
    const auto     ENDY   = h - 1;
    const auto     ENDX   = right ? (w - BORDERX) : BORDERX;
    auto           midx   = w / 2;
    auto           midy   = edges->height / 2;
    unsigned char* ptr    = (unsigned char*)edges->imageData;

    // show responses
    auto* votes = new int[lanes.size()];
    for (auto i    = 0; i < lanes.size(); i++)
        votes[i++] = 0;

    for (auto y = ENDY; y >= BEGINY; y -= SCAN_STEP)
    {
        std::vector< int > rsp;
        FindResponses(edges, midx, ENDX, y, rsp);

        if (rsp.size() > 0)
        {
            auto response_x = rsp[0]; // use first reponse (closest to screen center)

            float dmin  = 9999999;
            float xmin  = 9999999;
            auto  match = -1;
            for (auto j = 0; j < lanes.size(); j++)
            {
                // compute response point distance to current line
                float d = dist2line(cvPoint2D32f(lanes[j].p0.x, lanes[j].p0.y),
                                    cvPoint2D32f(lanes[j].p1.x, lanes[j].p1.y),
                                    cvPoint2D32f(response_x, y));

                // point on line at current y line
                auto xline    = (y - lanes[j].b) / lanes[j].k;
                auto dist_mid = abs(midx - xline); // distance to midpoint

                // pick the best closest match to line & to screen center
                if (match == -1 || (d <= dmin && dist_mid < xmin))
                {
                    dmin  = d;
                    match = j;
                    xmin  = dist_mid;
                    break;
                }
            }

            // vote for each line
            if (match != -1)
            {
                votes[match] += 1;
            }
        }
    }

    int bestMatch = -1;
    int mini      = 9999999;
    for (int i = 0; i < lanes.size(); i++)
    {
        int xline = (midy - lanes[i].b) / lanes[i].k;
        int dist  = abs(midx - xline); // distance to midpoint

        if (bestMatch == -1 || (votes[i] > votes[bestMatch] && dist < mini))
        {
            bestMatch = i;
            mini      = dist;
        }
    }

    if (bestMatch != -1)
    {
        Lane* best   = &lanes[bestMatch];
        float k_diff = fabs(best->k - side->k.get());
        float b_diff = fabs(best->b - side->b.get());

        bool update_ok = (k_diff <= K_VARY_FACTOR && b_diff <= B_VARY_FACTOR) || side->reset;

        printf("side: %s, k vary: %.4f, b vary: %.4f, lost: %s\n",
               (right ? "RIGHT" : "LEFT"),
               k_diff,
               b_diff,
               (update_ok ? "no" : "yes"));

        if (update_ok)
        {
            // update is in valid bounds
            side->k.add(best->k);
            side->b.add(best->b);
            side->reset = false;
            side->lost  = 0;
        }
        else
        {
            // can't update, lanes flicker periodically, start counter for partial reset!
            side->lost++;
            if (side->lost >= MAX_LOST_FRAMES && !side->reset)
            {
                side->reset = true;
            }
        }
    }
    else
    {
        printf("no lanes detected - lane tracking lost! counter increased\n");
        side->lost++;
        if (side->lost >= MAX_LOST_FRAMES && !side->reset)
        {
            // do full reset when lost for more than N frames
            side->reset = true;
            side->k.clear();
            side->b.clear();
        }
    }

    delete[] votes;
}

void processLanes(CvSeq* lines, IplImage* edges, IplImage* temp_frame)
{

    // classify lines to left/right side
    auto left  = std::vector< Lane >{};
    auto right = std::vector< Lane >{};

    for (auto i = 0; i < lines->total; i++)
    {
        auto line  = reinterpret_cast< CvPoint* >(cvGetSeqElem(lines, i));
        auto dx    = line[1].x - line[0].x;
        auto dy    = line[1].y - line[0].y;
        auto angle = atan2f(dy, dx) * 180 / CV_PI;

        if (fabs(angle) <= LINE_REJECT_DEGREES)
        { // reject near horizontal lines
            continue;
        }

        // assume that vanishing point is close to the image horizontal center
        // calculate line parameters: y = kx + b;
        dx     = (dx == 0) ? 1 : dx; // prevent DIV/0!
        auto k = dy / (float)dx;
        auto b = line[0].y - k * line[0].x;

        // assign lane's side based by its midpoint position
        int midx = (line[0].x + line[1].x) / 2;
        if (midx < temp_frame->width / 2)
        {
            left.push_back(Lane(line[0], line[1], angle, k, b));
        }
        else if (midx > temp_frame->width / 2)
        {
            right.push_back(Lane(line[0], line[1], angle, k, b));
        }
    }

    // show Hough lines
    for (int i = 0; i < right.size(); i++)
    {
        cvLine(temp_frame, right[i].p0, right[i].p1, CV_RGB(0, 0, 255), 2);
    }

    for (int i = 0; i < left.size(); i++)
    {
        cvLine(temp_frame, left[i].p0, left[i].p1, CV_RGB(255, 0, 0), 2);
    }

    processSide(left, edges, false);
    processSide(right, edges, true);

    // show computed lanes
    int x  = temp_frame->width * 0.55f;
    int x2 = temp_frame->width;
    cvLine(temp_frame,
           cvPoint(x, laneR.k.get() * x + laneR.b.get()),
           cvPoint(x2, laneR.k.get() * x2 + laneR.b.get()),
           CV_RGB(255, 0, 255),
           2);

    x  = temp_frame->width * 0;
    x2 = temp_frame->width * 0.45f;
    cvLine(temp_frame,
           cvPoint(x, laneL.k.get() * x + laneL.b.get()),
           cvPoint(x2, laneL.k.get() * x2 + laneL.b.get()),
           CV_RGB(255, 0, 255),
           2);
}

void runLoop(CvCapture* input_video)
{
    auto font = CvFont{};
    cvInitFont(&font, CV_FONT_VECTOR0, 0.25f, 0.25f);

    auto video_size = CvSize{};
    video_size.height =
        static_cast< int32_t >(cvGetCaptureProperty(input_video, CV_CAP_PROP_FRAME_HEIGHT));
    video_size.width =
        static_cast< int32_t >(cvGetCaptureProperty(input_video, CV_CAP_PROP_FRAME_WIDTH));

    const auto w = video_size.width;
    const auto h = video_size.height / 2;

    auto frame_size = cvSize(w, h);

    auto del_image = [](IplImage* ptr) {
        if (ptr)
        {
            cvReleaseImage(&ptr);
        }
    };

    auto temp_frame =
        std::shared_ptr< IplImage >(cvCreateImage(frame_size, IPL_DEPTH_8U, 3), del_image);

    auto grey  = std::shared_ptr< IplImage >(cvCreateImage(frame_size, IPL_DEPTH_8U, 1), del_image);
    auto edges = std::shared_ptr< IplImage >(cvCreateImage(frame_size, IPL_DEPTH_8U, 1), del_image);
    auto half_frame = std::shared_ptr< IplImage >(
        cvCreateImage(cvSize(video_size.width / 2, video_size.height / 2), IPL_DEPTH_8U, 3),
        del_image);

    auto houghStorage = cvCreateMemStorage(0);
    auto haarStorage  = cvCreateMemStorage(0);
    auto cascade      = reinterpret_cast< CvHaarClassifierCascade* >(cvLoad(
        "/Users/jaimerios/Development/GitHub/opencv-lane-vehicle-track_master/bin/haar/cars3.xml"));
    assert(cascade && "Error found");

    // cvSetCaptureProperty(input_video, CV_CAP_PROP_POS_FRAMES, current_frame);
    auto       key_pressed = 0;
    const auto escape_key  = 27;
    while (key_pressed != escape_key)
    {
        const auto frame = cvQueryFrame(input_video);
        if (frame == nullptr)
        {
            fprintf(stderr, "Error: null frame received\n");
            return;
        }

        cvPyrDown(frame, half_frame.get(), CV_GAUSSIAN_5x5); // Reduce the image by 2
        // cvCvtColor(temp_frame, grey, CV_BGR2GRAY); // convert to grayscale

        // we're interested only in road below horizont - so crop top image portion off
        crop(frame,
             temp_frame.get(),
             cvRect(0, frame_size.height, frame_size.width, frame_size.height));
        cvCvtColor(temp_frame.get(), grey.get(), CV_BGR2GRAY); // convert to grayscale

        // Perform a Gaussian blur ( Convolving with 5 X 5 Gaussian) & detect edges
        cvSmooth(grey.get(), grey.get(), CV_GAUSSIAN, 5, 5);
        cvCanny(grey.get(), edges.get(), CANNY_MIN_TRESHOLD, CANNY_MAX_TRESHOLD);

        // do Hough transform to find lanes
        const auto rho   = 1;
        const auto theta = CV_PI / 180;
        auto       lines = cvHoughLines2(edges.get(),
                                   houghStorage,
                                   CV_HOUGH_PROBABILISTIC,
                                   rho,
                                   theta,
                                   HOUGH_TRESHOLD,
                                   HOUGH_MIN_LINE_LENGTH,
                                   HOUGH_MAX_LINE_GAP);

        processLanes(lines, edges.get(), temp_frame.get());

        // process vehicles
        vehicleDetection(half_frame.get(), cascade, haarStorage);
        drawVehicles(half_frame.get());
        cvShowImage("Half-frame", half_frame.get());
        cvMoveWindow("Half-frame", half_frame->width * 2 + 10, 0);

        // show middle line
        cvLine(temp_frame.get(),
               cvPoint(frame_size.width / 2, 0),
               cvPoint(frame_size.width / 2, frame_size.height),
               CV_RGB(255, 255, 0),
               1);

        cvShowImage("Grey", grey.get());
        cvShowImage("Edges", edges.get());
        cvShowImage("Color", temp_frame.get());

        cvMoveWindow("Grey", 0, 0);
        cvMoveWindow("Edges", 0, frame_size.height + 25);
        cvMoveWindow("Color", 0, 2 * (frame_size.height + 25));

        key_pressed = cvWaitKey(5);
    }

    cvReleaseHaarClassifierCascade(&cascade);
    cvReleaseMemStorage(&haarStorage);
    cvReleaseMemStorage(&houghStorage);
}

int main(void)
{
#ifdef USE_VIDEO
    auto input_video = cvCreateFileCapture(
        "/Users/jaimerios/Development/GitHub/opencv-lane-vehicle-track_master/bin/road.avi");
#else
    auto input_video = std::shared_ptr< CvCapture >(cvCreateCameraCapture(0), [](CvCapture* ptr) {
        if (ptr != nullptr)
        {
            cvReleaseCapture(&ptr);
        }
    });

#endif

    if (input_video == nullptr)
    {
        fprintf(stderr, "Error: Can't open video\n");
        return -1;
    }
    runLoop(input_video.get());
}
