function showPic(index, test)

	load data.mat
	load result.mat

	if test == 0
		img = reshape( train_x(index,:), 96, 96);
		answer = train_y(index,:);
		points = reshape(answer, 2, 15);
		new_img = img/256;
	else
		img = reshape( test_x(index,:), 96, 96);
		answer = test_y(index,:);
		points = reshape(answer, 2, 15);
		new_img = img;
	end
	
	for i=1:15
		new_img( round(points(1,i)), round(points(2,i)) ) = 1;
		new_img( round(points(1,i))+1, round(points(2,i)) ) = 1;
		new_img( round(points(1,i))-1, round(points(2,i)) ) = 1;
		new_img( round(points(1,i)), round(points(2,i))+1 ) = 1;
		new_img( round(points(1,i)), round(points(2,i))-1 ) = 1;
	end

	imwrite(new_img, 'test.bmp')
