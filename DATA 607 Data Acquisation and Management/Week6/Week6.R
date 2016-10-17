library(jsonlite)
library(XML)
library(htmltab)


json_file <- "C:/data/book.json"
json_data <- fromJSON(paste(readLines(json_file), collapse=""))
jsonDF <- data.frame(json_data)
View(jsonDF)

rawXML <- xmlParse("C:/data/book.xml")
xmlDF <- xmlToDataFrame(rawXML)
xmlDF$Amazon
View(xmlDF)

html_data <- htmltab(doc = "C:/data/book.html")
View(html_data)





