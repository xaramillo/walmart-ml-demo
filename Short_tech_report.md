# Short tech report

During the design of the present solution, I faced multiple challenges, being on the top 3 considerations:

1. **Performance**: I tested diferent modeling algorithm based on the classification solution that I want to implement, but using Codespaces as the demo takes a lot of time because of the commputer capabilities for this amount of data
2. **Better feature recognition**: Due to time constrain, I decided to implement classification based on the telemetry as the base dataset with failures labels attached, perhaps we could consider in the near future to recognize better features to lower the model training time and resources usage, we should consider the other datasets as well
3. **Modularity**: For the sake of registering and running a modern pipeline implementation, I modularized the ML pipeline into different scripts. This helps to have cleaner code and easy to maintain and add more features, in my mind this has the chance to be used on multiple scenarios and for multiple purposes.