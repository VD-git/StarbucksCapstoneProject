# StarbucksCapstoneProject
Udacity Machine Learning Engineer AWS Capstone Project

# Problem Overview
The data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be:
- Merely an **advertisement** for a drink or an actual offer such as a **discount**;
- **BOGO** (buy one get one free);
- Some users might **not receive** any offer during certain weeks.


Not all users receive the same offer, and that is the challenge to solve with this data set in order to improve the customer journey.


A key challenge too, it is not only discovering which one is the best offer to be sent, sometimes the person is static to it, being the goal to discover whether a client will buy in or not the offer. A group of individuals are few times assiduous customers, but even receiving the offer he or she does not use it, or do not even look at it, using it sometimes without realizing, only discovering the discount or the promo at the cashier.


The main goal of the offer is to avoid churn and keep clients around, when the churn probability increases, but if a customer is “uncorrelated” with offers and promotions, even being a premium client or not, does not add value to the business sending a promotion, since it has a cost aggregate to it.


A task within the project is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products. Even gathering outside data if it is pertinent in order to achieve better performance for the model.


The possible customers' journeys are described below, what it is sought are the one with the green arrow. It also not wanted to send an offer, the customer receives it, buy but have not viewed. That means that the client used the discount, but would buy anyway.


![Possibilities of customer journey](/images/business_metrics.png "Possibilities of customer journey")


# Requierements
- Inside the sagemaker notebook instance, the kernel that matched with what it was requiered was the one of tensorflow;
- Few libraries were needed to be install in order to go on with the whole analysis as shown below.

## **kernel**
conda_tensorflow2_p310

## **libraries**
lightgbm==4.1.0

sagemaker==2.197.0

ydata-profiling==4.6.2

shap==0.43.0

# Acknowledgements
Dataset was provided by [Starbucks](https://www.starbucks.com/).

