<!-- This is the markdown template for the final project of the Building AI course, 
created by Reaktor Innovations and University of Helsinki. 
Copy the template, paste it to your GitHub README and edit! -->

# Project Title

Final project for the Building AI course

## Summary

Describe briefly in 2-3 sentences what your project is about. About 250 characters is a nice length! 


## Background

With the rapid growth of flight flow, the workload of controllers is increasing daily, and
handling flight conflicts is the main workload.  To cope with the continuous increase
of flight flow and reasonably reduce the controller’s workload,
intelligent conflict resolution methods should be studied to
provide the controller with decision-making support in line
with actual operation. In our project, a Deep Reinforcement Learning algorithm is proposed to resolve aircraft flight conflict with high solving efficiency. Conflict
resolution algorithm based on DRL has great advantages in solution speed



## How is it used?

Describe the process of using the solution. In what kind situations is the solution needed (environment, time, etc.)? Who are the users, what kinds of needs should be taken into account?

Images will make your README look nice!
Once you upload an image to your repository, you can link link to it like this (replace the URL with file path, if you've uploaded an image to Github.)
![Cat](https://upload.wikimedia.org/wikipedia/commons/5/5e/Sleeping_cat_on_her_back.jpg)

If you need to resize images, you have to use an HTML tag, like this:
<img src="https://upload.wikimedia.org/wikipedia/commons/5/5e/Sleeping_cat_on_her_back.jpg" width="300">

This is how you create code examples:
```
def main():
   countries = ['Denmark', 'Finland', 'Iceland', 'Norway', 'Sweden']
   pop = [5615000, 5439000, 324000, 5080000, 9609000]   # not actually needed in this exercise...
   fishers = [1891, 2652, 3800, 11611, 1757]

   totPop = sum(pop)
   totFish = sum(fishers)

   # write your solution here

   for i in range(len(countries)):
      print("%s %.2f%%" % (countries[i], 100.0))    # current just prints 100%

main()
```


## Data sources and AI methods. Simulation

At first we decided to use real data for the simulation, but there was not always a conflict in this dataset and it would be problematic for our model. Because of this, we ourselves generate conflicts between aircraft. Aircrafts are randomly created and conflicting aircrafts to these aircrafts. In addition to conflicting pairs, conventional aircraft are also created to simulate a real environment.


## Challenges

What does your project _not_ solve? Which limitations and ethical considerations should be taken into account when deploying a solution like this?

## What next?

How could your project grow and become something even more? What kind of skills, what kind of assistance would you  need to move on? 


## Acknowledgments

* list here the sources of inspiration 
* do not use code, images, data etc. from others without permission
* when you have permission to use other people's materials, always mention the original creator and the open source / Creative Commons licence they've used
  <br>For example: [Sleeping Cat on Her Back by Umberto Salvagnin](https://commons.wikimedia.org/wiki/File:Sleeping_cat_on_her_back.jpg#filelinks) / [CC BY 2.0](https://creativecommons.org/licenses/by/2.0)
* etc