#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
//#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/subscriber_filter.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

class DisparityMapCalculator
{
public:
  DisparityMapCalculator(const std::string &img_transport);
  ~DisparityMapCalculator();
  enum eMethod
  {
    eMethod_BM = 1,
    eMethod_SGBM
  } m_method;

private:
  image_transport::SubscriberFilter m_img_left_sub, m_img_right_sub;
  image_transport::Publisher m_disp_pub;
  typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image> ExactPolicy;
  typedef message_filters::Synchronizer<ExactPolicy> ExactSync;
  boost::shared_ptr<ExactSync> m_exact_sync;
  ros::Time m_last_update_time;

  cv::Ptr<cv::StereoBM> m_bm;
  cv::Ptr<cv::StereoSGBM> m_sgbm;
  int m_num_disparities;

  void OnStereoImage(const sensor_msgs::ImageConstPtr &img_msg_left, const sensor_msgs::ImageConstPtr &img_msg_right);
};

DisparityMapCalculator::DisparityMapCalculator(const std::string &img_transport) : m_num_disparities(128)
{
  ros::NodeHandle nh;
  std::string stereo_ns = nh.resolveName("stereo");
  std::string left_img_topic = ros::names::clean(stereo_ns + "/left/" + nh.resolveName("image_rect"));
  std::string right_img_topic = ros::names::clean(stereo_ns + "/right/" + nh.resolveName("image_rect"));
  ROS_INFO("Subscribed to:\n\t* %s\n\t* %s", left_img_topic.c_str(), right_img_topic.c_str());

  image_transport::ImageTransport it(nh);
  m_img_left_sub.subscribe(it, left_img_topic, 10, img_transport);
  m_img_right_sub.subscribe(it, right_img_topic, 10, img_transport);

  ros::NodeHandle local_nh("~");

  int queue_size;
  local_nh.param("queue_size", queue_size, 10);

  int method = 1;
  local_nh.param("method", method, 1);
  m_method = static_cast<eMethod>(method);
  if (m_method == eMethod_BM)
  {
    m_bm = cv::StereoBM::create(16, 9);
    m_bm->setPreFilterCap(31);
    m_bm->setBlockSize(9);
    m_bm->setMinDisparity(0);
    m_bm->setNumDisparities(m_num_disparities);
    m_bm->setTextureThreshold(10);
    m_bm->setUniquenessRatio(15);
    m_bm->setSpeckleWindowSize(100);
    m_bm->setSpeckleRange(32);
    m_bm->setDisp12MaxDiff(1);
  }
  else if (m_method == eMethod_SGBM)
  {
    m_sgbm = cv::StereoSGBM::create(0, 16, 3);
    m_sgbm->setPreFilterCap(63);
    int sgbmWinSize = 9;
    m_sgbm->setBlockSize(sgbmWinSize);
    int cn = 3;
    m_sgbm->setP1(8 * cn * sgbmWinSize * sgbmWinSize);
    m_sgbm->setP2(32 * cn * sgbmWinSize * sgbmWinSize);
    m_sgbm->setMinDisparity(0);
    m_sgbm->setNumDisparities(m_num_disparities);
    m_sgbm->setUniquenessRatio(10);
    m_sgbm->setSpeckleWindowSize(100);
    m_sgbm->setSpeckleRange(32);
    m_sgbm->setDisp12MaxDiff(1);
    m_sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
  }

  m_exact_sync.reset(new ExactSync(ExactPolicy(queue_size), m_img_left_sub, m_img_right_sub));
  m_exact_sync->registerCallback(boost::bind(&DisparityMapCalculator::OnStereoImage, this, _1, _2));

  std::string disparity_topic = "disparity/disparity_image";
  local_nh.getParam("disparity_topic", disparity_topic);
  m_disp_pub = it.advertise(disparity_topic, 1);
  ROS_INFO_STREAM("Advertised on topic " << disparity_topic);
}

DisparityMapCalculator::~DisparityMapCalculator()
{
}

static void ConvertToGray(cv::Mat const *const bgr, cv::Mat *gray)
{
  cv::cvtColor(*bgr, *gray, cv::COLOR_BGR2GRAY);
}

void DisparityMapCalculator::OnStereoImage(const sensor_msgs::ImageConstPtr &img_msg_left, const sensor_msgs::ImageConstPtr &img_msg_right)
{
  const ros::Time &timestamp = img_msg_left->header.stamp;
  if (m_last_update_time > timestamp)
  {
    ROS_WARN("Images with older time stamps recieved. Will be ignored.");
    return;
  }

  cv_bridge::CvImageConstPtr img_ptr_l = cv_bridge::toCvShare(img_msg_left, "bgr8");
  cv_bridge::CvImageConstPtr img_ptr_r = cv_bridge::toCvShare(img_msg_right, "bgr8");
  ROS_WARN_COND(img_ptr_l->image.empty(), "LEFT IMAGE IS EMPTY!!");
  ROS_WARN_COND(img_ptr_r->image.empty(), "RIGHT IMAGE IS EMPTY!!");
  ROS_ASSERT(img_msg_left->width == img_msg_right->width);
  ROS_ASSERT(img_msg_left->height == img_msg_right->height);

  // Convert to grayscale in parallel
  cv::Mat left_gray, right_gray;
  std::thread right_conv_th(ConvertToGray, &(img_ptr_r->image), &right_gray);
  cv::cvtColor(img_ptr_l->image, left_gray, cv::COLOR_BGR2GRAY);
  right_conv_th.join();

  cv::Mat disp, disp8;
  if (m_method == eMethod_BM)
  {
    m_bm->compute(left_gray, right_gray, disp);
  }
  else if (m_method == eMethod_SGBM)
  {
    m_sgbm->compute(left_gray, right_gray, disp);
  }
  disp.convertTo(disp8, CV_8U, 255 / (m_num_disparities * 16.));

  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", disp8).toImageMsg();
  msg->header.stamp = timestamp;
  msg->header.frame_id = img_msg_left->header.frame_id;
  m_disp_pub.publish(msg);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "DisparityMapCalculator");
  DisparityMapCalculator d("raw");
  ros::spin();
  return 0;
}
