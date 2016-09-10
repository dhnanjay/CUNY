CREATE DATABASE `movies` /*!40100 DEFAULT CHARACTER SET utf8 */;

CREATE TABLE `ratings` (
  `CriticName` varchar(45) NOT NULL COMMENT 'Critic Name',
  `MovieName` varchar(45) NOT NULL COMMENT 'Movie Name',
  `Year` varchar(4) DEFAULT NULL COMMENT 'Year in which Movie was released',
  `Genre` varchar(15) DEFAULT NULL COMMENT 'Genre: Action| Comedy| Crime| Romance| Thriller',
  `Rating` int(11) NOT NULL COMMENT 'Rating between 1 to 5 where 5  is the highest rating and 1 lowest',
  PRIMARY KEY (`CriticName`,`MovieName`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='Table for storing User Ratings for the Movies';

LOAD DATA INFILE 'c:/data/ratings.csv' INTO TABLE movies.ratings
FIELDS TERMINATED BY ','
ENCLOSED BY  '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES ;

SELECT * FROM movies.ratings;




