library(shiny)
library(ggplot2)
library(dplyr)

cdc <- read.csv("cleaned-cdc-mortality-1999-2010-2.csv")

ui <- fluidPage(
  titlePanel("Mortality Rates by State"),
  sidebarLayout(
    sidebarPanel(paste0("Please select mortality cause and year to see comparison of rates between states.",
                        "National average is shown with a dashed line."),
                 br(),
                 uiOutput("causeOutput"),
                 radioButtons("sortInput", "Sort Order",
                              choices = c("Descending", "Ascending"),
                              selected = "Descending", inline = FALSE),
                 uiOutput("yearOutput")
    ),
    mainPanel(plotOutput("barplot")
    )
  )
)

server <- function(input, output) {
  output$causeOutput <- renderUI({
    selectInput("causeInput", "Cause",
                sort(unique(cdc$ICD.Chapter)),
                selected = "Neoplasms")
  })
  output$yearOutput <- renderUI({
    selectInput("yearInput", "Year",
                sort(unique(cdc$Year)),
                selected = "2010")
  })
  cdc_filt <- reactive({
    if (is.null(input$causeInput)) {
      return(NULL)
    }  
    cdc %>%
      filter(ICD.Chapter == input$causeInput,
             Year == input$yearInput)
  })
  national_avg <- reactive({
    if (is.null(input$causeInput)) {
      return(NULL)
    }
    national <- cdc_filt() %>% 
      group_by(Year) %>%
      summarize(totalDeath = sum(Deaths), totalPop = sum(Population)) %>%
      mutate(Rate = (100000*totalDeath)/totalPop)
    national$Rate[1]
  })
  
  output$barplot <- renderPlot({
    if (is.null(cdc_filt())) {
      return()
    }
    
    cdc_plot <- cdc_filt()
    cdc_sort <- (input$sortInput == "Ascending") 
    cdc_plot$State <- factor(cdc_plot$State, levels = cdc_plot[order(cdc_plot$Crude.Rate, decreasing = cdc_sort), "State"])
    
    ggplot(data = cdc_plot, aes(x = State, y = Crude.Rate)) + 
      geom_bar(aes(fill = Crude.Rate), stat="identity") + 
      geom_text(aes(label=Crude.Rate), hjust=-0.1, vjust=0.4) +
      geom_hline(yintercept = national_avg(), linetype = "dashed") + 
      coord_flip() + 
      labs(x = "", y = "") +
      ggtitle(paste(input$causeInput, "/", input$yearInput)) +
      theme(panel.background = element_blank(),
            axis.ticks = element_blank(),
            axis.text.x = element_blank(),
            axis.text.y = element_text(size = 12, margin = margin(r=-20)),
            legend.position="none")
    
  }, height = 800)
}
shinyApp(ui = ui, server = server)
