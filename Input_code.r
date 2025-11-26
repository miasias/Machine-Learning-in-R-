###############################################
# Application of the tuned Random forest model#
###############################################

#This code defines functions to safely collect and validate user input, converts the input into a data.table compatible
#with mlr3, and uses the tuned Random Forest model to generate a heart-disease prediction for the new person.

#Run the code and then insert your own values into the console for every predictor. 
#The model will then predict the risk for a cardiovascular disease.

library(data.table)

# Create function to test integer input with optional bounds
read_int <- function(inputs, min = NULL, max = NULL) {
  repeat {
    # Read user input
    x <- readline(inputs)
    
    # Check if the input contains only digits (no letters or decimals)
    if (!grepl("^\\d+$", x)) {
      cat("Please enter an integer number.\n")
      next
    }
    # Convert input to an integer
    val <- as.integer(x)
    
    # Check optional minimum allowed value
    if (!is.null(min) && val < min) {
      cat("Value must be >=", min, "\n")
      next
    }
    # Check optional maximum allowed value
    if (!is.null(max) && val > max) {
      cat("Value must be <=", max, "\n")
      next
    }
    
    return(val)
  }
}


# Create a function to test numeric input with optional bounds
read_num <- function(inputs, min = NULL, max = NULL) {
  repeat {
    # Read user input
    x <- readline(inputs)
    # Try converting to numeric (allows decimals) and prevent Warning messages
    val <- suppressWarnings(as.numeric(x))
    if (is.na(val)) {
      cat("Please enter a numeric value.\n")
      next
    }
    # Minimum bound check
    if (!is.null(min) && val < min) {
      cat("Value must be >=", min, "\n")
      next
    }
    # Maximum bound check
    if (!is.null(max) && val > max) {
      cat("Value must be <=", max, "\n")
      next
    }
    return(val)
  }
}

# Create a function that asks User for input and creates a dataframe with the input
create_person <- function() {
  
  cat("Please enter the following values:\n")
  
  # Use function read_int to collect validated integer inputs from the user
  age              <- read_int("Age (years): ", min = 1)
  gender           <- read_int("Gender (0 = Female, 1 = Male): ", min = 0, max = 1)
  chestpain        <- read_int("Chest pain type (0–3): ", min = 0, max = 3)
  restingBP        <- read_int("Resting blood pressure (mmHg): ", min = 0)
  serumcholestrol  <- read_int("Serum cholesterol (mg/dl): ", min = 0)
  
  # Get fasting blood sugar as a numeric value from the User
  fbs_value        <- read_num("Fasting blood sugar (mg/dl): ", min = 0)
  # Automatically convert the input into a binary category
  fastingbloodsugar <- ifelse(fbs_value >= 120, 1L, 0L)
  
  # Collect more inputs from the user and validate them using the defined functions 
  restingrelectro  <- read_int("Resting electrocardiographic results (0–2): ", min = 0, max = 2)
  maxheartrate     <- read_int("Maximum heart rate achieved: ", min = 0)
  exerciseangia    <- read_int("Exercise-induced angina (0 = No, 1 = Yes): ", min = 0, max = 1)
  oldpeak          <- read_num("Oldpeak: ", min = 0)
  noofmajorvessels <- read_int("Number of major vessels (0–3): ", min = 0, max = 3)
  
  # Build a dataframe containing all user inputs in the same structure as the training data
  person <- data.frame(
    # A Dummy ID that is used to preserve structure
    patientid        = 999999,
    age              = age,
    gender           = gender,
    chestpain        = chestpain,
    restingBP        = restingBP,
    serumcholestrol  = serumcholestrol,
    fastingbloodsugar = fastingbloodsugar,
    restingrelectro  = restingrelectro,
    maxheartrate     = maxheartrate,
    exerciseangia    = exerciseangia,
    oldpeak          = oldpeak,
    noofmajorvessels = noofmajorvessels
  )
  
  # Return the dataframe with the inputs of the User
  return(person)
}

# Call the function create_person
predict_new_person <- function() {
  # Call the function create_person
  new_person <- create_person()
  # Convert User input into a data table format
  new_person_dt <- as.data.table(new_person)
  # Using the created data table we make a prediction using the tuned model
  prediction_new <- rf_tuned$predict_newdata(
    task = task_data,              # manual task passed here
    newdata = new_person_dt
  )
  
  cat("\n---- Prediction for new person ----\n")
  # Print and return the prediction 
  print(prediction_new)
  return(prediction_new)
}

# Call the function above as a test
  
  predict_new_person()

