# Vision Algorithms for Mobile Robotics at UZH

Code for the exercises and mini-project of the course [Vision Algorithms for Mobile Robotics at UZH](https://rpg.ifi.uzh.ch/teaching.html)

The course is thought by [Prof. Dr. Davide Scaramuzza](https://rpg.ifi.uzh.ch/people_scaramuzza.html)

## Possible improvements

- [ ] Consistent naming
  - [ ] Python is usually snake case rather than camel case
- [ ] Logging
- [ ] Factoring out functions / using utils
- [ ] When an exercise has not been started, running the program should not throw an error but print a helpful message instead. This lets students
know, your setup is fine (vs an error: is there a problem with how I run the code?)
- [ ] Add flags for when to show output, otherwise have to close images every time the code is run
  - [ ] Have a flag to display intermediate results (blurred, DoGs, with corners detected, ...)
- [ ] Add some tests or assert statements
- [ ] Standard docstrings (reST or Google)
- [ ] Points are sometimes stored as (2, N), sometimes as (N, 2). Why not always store them in the same format for consistency?
- [ ] Ex 7, 000001.jpg is missing in /data
