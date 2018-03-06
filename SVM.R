install.packages('e1071', dependencies = TRUE)




library (e1071)
letters = read.csv("/Users/jeremy/Downloads/letters.csv")
vowel = read.csv("/Users/jeremy/Downloads/vowel.csv")

runVowel <- function(g, c){
allRows <- 1:nrow(vowel)
testRows <- sample(allRows, trunc(length(allRows) * 0.3))
vowelTest <- vowel[testRows,]
vowelTrain <- vowel[-testRows,]
model <- svm(Class~., data = vowelTrain, kernel = "radial", gamma = g, cost = 10)

prediction <- predict(model, vowelTest[,-ncol(vowel)])

confusionMatrix <- table(pred = prediction, true = vowelTest$Class)

agreement <- prediction == vowelTest$Class
accuracy <- prop.table(table(agreement))

return(accuracy)
}

# ------------------------------------------------------------------------------------

runLetters <- function(g, c){
  allRows <- 1:nrow(letters)
  testRows <- sample(allRows, trunc(length(allRows) * 0.3))
  lettersTest <- letters[testRows,]
  lettersTrain <- letters[-testRows,]
  lettersTrain <- sample(1:nrow(lettersTrain), trunc(length(1:nrow(lettersTrain)) * 0.01))
  lettersTrain <- letters[lettersTrain,]
  model <- svm(letter~., data = lettersTrain, kernel = "radial", gamma = g, cost = c)
  
  prediction <- predict(model, lettersTest[,-1])
  confusionMatrix <- table(pred = prediction, true = lettersTest$letter)
  
  agreement <- prediction == lettersTest$letter
  accuracy <- prop.table(table(agreement))
  
  return(accuracy)
}

vowelMatrix <- function(gamma){
  invisible(mapply(runVowel, g = gamma , c = 2^(seq(-15,15,2))))
}

letterMatrix <- function(gamma){
  invisible(mapply(runLetters, g = gamma , c = 2^(seq(-15,15,2))))
}

mapply(vowelMatrix, g = 2^(seq(-15,15,2)))
mapply(letterMatrix, g = 2^(seq(-15,15,2)))