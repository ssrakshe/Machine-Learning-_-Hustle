# Machine-Learning #Hustle

Agenda:<br>
delved into the details of gradient descent for a single variable.<br>
developed a routine to compute the gradient<br>
visualized what the gradient is<br>
completed a gradient descent routine<br>
utilized gradient descent to find parameters<br>
examined the impact of sizing the learning rate<br>

Housing price prediction based on area of house.
Problem Statement
Let's use the same two data points as before - a house with 1000 square feet sold for $300,000 and a house with 2000 square feet sold for $500,000.
![LRGDC1](https://github.com/ssrakshe/Machine-Learning-_-Hustle/assets/66088285/9cc470ae-0a31-44b5-845a-c2e3ebff2296)

![LRGDC2](https://github.com/ssrakshe/Machine-Learning-_-Hustle/assets/66088285/3db30e65-9868-4fc2-8588-9daf3c8a2575)

The Gradient Descent Algorithm
Gradient descent is an iterative optimization algorithm to find the minimum of a function. Here that function is our Cost Function.

Imagine a valley and a person with no sense of direction who wants to get to the bottom of the valley. He goes down the slope and takes large steps when the slope is steep and small steps when the slope is less steep. He decides his next position based on his current position and stops when he gets to the bottom of the valley which was his goal.
Let’s try applying gradient descent to m and c and approach it step by step:

Initially let m = 0 and c = 0. Let L be our learning rate. This controls how much the value of m changes with each step. L could be a small value like 0.0001 for good accuracy.
Calculate the partial derivative of the loss function with respect to m, and plug in the current values of x, y, m and c in it to obtain the derivative value D.

Derivative with respect to m
Dₘ is the value of the partial derivative with respect to m. Similarly lets find the partial derivative with respect to c, Dc :


Derivative with respect to c
3. Now we update the current value of m and c using the following equation:


4. We repeat this process until our loss function is a very small value or ideally 0 (which means 0 error or 100% accuracy). The value of m and c that we are left with now will be the optimum values.

Now going back to our analogy, m can be considered the current position of the person. D is equivalent to the steepness of the slope and L can be the speed with which he moves. Now the new value of m that we calculate using the above equation will be his next position, and L×D will be the size of the steps he will take. When the slope is more steep (D is more) he takes longer steps and when it is less steep (D is less), he takes smaller steps. Finally he arrives at the bottom of the valley which corresponds to our loss = 0.
Now with the optimum value of m and c our model is ready to make predictions !
