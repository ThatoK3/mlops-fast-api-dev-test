CREATE DATABASE stroke_predictions;

CREATE USER 'pred_user'@'%' IDENTIFIED BY 'pred_user_pass';
GRANT ALL PRIVILEGES ON stroke_predictions.* TO 'pred_user'@'%';
FLUSH PRIVILEGES;
