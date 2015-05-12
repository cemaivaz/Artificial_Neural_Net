dim = 20;
num_obstacles = 19;
num_episodes = 2000;
plot_freq = 200; % every $plot_freq images are plotted
save_maze = 0; % 0 = false, 1 = true
img_dir = 'images'; % image directory; where to save images
%initialize parameters
epsilon = .75;
gamma = .75;
alpha = .1;
lambda = .9;
mu = .999;
%initialize goal position
goalX = randi( dim ) - .5;
goalY = randi( dim ) - .5;
%goalX = 13.5
%goalY = 12.5
%initialize obstacles to zeros
obstaclesX = zeros( 1, num_obstacles );
obstaclesY = zeros( 1, num_obstacles );
%add goal to obstacles so randomly generated obstacles aren't in the goal
obstaclesX(1) = goalX;
obstaclesY(1) = goalY;
%randomly generate obstacles
for i=2:num_obstacles
 newObX = randi( dim ) - .5;
 newObY = randi( dim ) - .5;
 while Check_obstacle( newObX, newObY, obstaclesX, obstaclesY )
 newObX = randi( dim ) - .5;
 newObY = randi( dim ) - .5;
 end
 obstaclesX(i) = newObX;
 obstaclesY(i) = newObY;
end
%remove goal from obstacles
obstaclesX = obstaclesX(2:end);
obstaclesY = obstaclesY(2:end);
%initialize Q(s,a) arbitrarily
%Q = rand( [dim, dim, 4] ) * .25;
Q = zeros( dim, dim, 4 );
Q( (obstaclesX+.5), (obstaclesY+.5), : ) = 0;
%eligability trace
%et = zeros( dim, dim, 4 );
for i=1:num_episodes
 %begin an episode

 %initialize start state -- don't run into obstacles and be a bit from
 %the goal
 X = randi(dim) - .5;
 Y = randi(dim) - .5;
 while (abs(X-goalX) < 2 ) || Check_obstacle(X,Y,obstaclesX,obstaclesY) || (abs(Y-goalY) < 2 )
 X = randi(dim) - .5;
 Y = randi(dim) - .5;
 end
 startX = X;
 startY = Y;

 %these matricies will hold the x,y positions traveled
 xmat = 0;
 ymat = 0;
 steps = 0;

 %repeat for each step
 while( 1 )
 %save the number of steps it has taken
 steps = steps + 1;

 %save the x and y positions
 xmat( steps ) = X;
 ymat( steps ) = Y;

 %choose action based on Q using epsilon-greedy
 rannum = rand();
 [val,ind] = max(Q(X+.5,Y+.5,:));
 if rannum > epsilon
 %take greedy
 action = ind;
 else
 %take non-greedy
 action = randi(4);
 while action == ind
 action = randi(4);
 end
 end
 %take action a, observe r,s'
 newX = X;
 newY = Y;
 switch action
 case 1
 newY = Y + 1; %up
 if Check_obstacle( X, newY, obstaclesX, obstaclesY )
 newY = Y;
 end
 case 2
 newX = X + 1; %right
 if Check_obstacle( newX, Y, obstaclesX, obstaclesY )
 newX = X;
 end
 case 3
 newX = X - 1; %left
 if Check_obstacle( newX, Y, obstaclesX, obstaclesY )
 newX = X;
 end
 case 4
 newY = Y - 1; %down
 if Check_obstacle( X, newY, obstaclesX, obstaclesY )
 newY = Y;
 end
 end

 %go back if it knocks you off the map
 if newX > dim || newX < 0
 newX = X;
 end
 if newY > dim || newY < 0
 newY = Y;
 end

 %only reward for hitting goal
 if newX == goalX && newY == goalY
 reward = 1;
 else
 reward = 0;
 end

 %Q-learning
 [val,next_act] = max(Q(newX+.5,newY+.5,:));
 %et( X+.5, Y+.5, action ) = et( X+.5, Y+.5, action ) + 1;
 %Q(X+.5,Y+.5,action) = Q(X+.5,Y+.5,action) +
et(X+.5,Y+.5,action)*alpha*( reward + (gamma*val)-Q(X+.5,Y+.5,action) );
 Q(X+.5,Y+.5,action) = Q(X+.5,Y+.5,action) + alpha*( reward + (gamma*val)-Q(X+.5,Y+.5,action) );

 %decay eligibility trace
 %et = gamma*lambda*et;

 %update the state
 X = newX;
 Y = newY;
 if X == goalX && Y == goalY
 break;
 end
 end

 %decay epsilon with time
 epsilon = epsilon*mu;

 %print out maze
 if( ~mod( (i-1), plot_freq ) )
 if( save_maze )
 map = figure( 'Visible', 'off' );
 else
 figure;
 end

 %set up grid to plot Q(s,a)
 gridX = repmat( transpose(.5:1:(dim-.5)), 1, dim );
 gridY = transpose( gridX );
 u = zeros( dim, dim );
 v = zeros( dim, dim );
 for a=1:dim
 for b=1:dim
 [val,ind] = max( Q(a,b,:) );
 switch ind
 case 1
 u(a,b) = 0;
 v(a,b) = val;
 case 2
 u(a,b) = val;
 v(a,b) = 0;
 case 3
 u(a,b) = -val;
 v(a,b) = 0;
 case 4
 u(a,b) = 0;
 v(a,b) = -val;
 end
 end
 end

 quiver( gridX, gridY, u, v, 0 ); %that's right, quiver
 hold on;

 %plot where we've been, the goal, start, and obstacles
 plot( xmat, ymat, 'ko' ); %black circles
 plot( obstaclesX, obstaclesY, 'ks','MarkerSize', 10, 'MarkerFaceColor', 'k' ); %black square
 plot( startX, startY, 'bd', 'MarkerSize', 10, 'MarkerFaceColor', 'b'); %blue diamond
 plot( goalX, goalY, 'bp','MarkerSize', 14, 'MarkerFaceColor', 'b' );
%blue pentagram
 grid on;

 axis( [0 dim 0 dim] );
 set( gca, 'YTick', 0:1:dim );
 set( gca, 'XTick', 0:1:dim );
 set( gca, 'GridLineStyle', '-' );
 
 if( save_maze )
 filename = sprintf( '%s/image_%d.ppm', img_dir, i );
 fprintf( 'saving %s...', filename );
 print( map, '-dppm', '-r200', filename );
 fprintf( 'done.\n' );
 close( map );
 else
 drawnow;
 end
 end
end
sum( steps ) / size( steps, 1 )
%print the final image
if( save_maze )
 map = figure( 'Visible', 'off' );
else
 figure;
end
%set up grid to plot Q(s,a)
 gridX = repmat( transpose(.5:1:(dim-.5)), 1, dim );
 gridY = transpose( gridX );
 u = zeros( dim, dim );
 v = zeros( dim, dim );
for a=1:dim
 for b=1:dim
 [val,ind] = max( Q(a,b,:) );
 switch ind
 case 1
 u(a,b) = 0;
 v(a,b) = val;
 case 2
 u(a,b) = val;
 v(a,b) = 0;
 case 3
 u(a,b) = -val;
 v(a,b) = 0;
 case 4
 u(a,b) = 0;
 v(a,b) = -val;
 end
 end
end
%output Q(s,a) using quiver
 quiver( gridX, gridY, u, v, 0 );
 hold on;
%plot where we've been, the goal, start, and obstacles
 plot( xmat, ymat, 'ko' ); %black circles
 plot( obstaclesX, obstaclesY, 'ks','MarkerSize', 10, 'MarkerFaceColor', 'k'); %black square
 plot( startX, startY, 'bd', 'MarkerSize', 10, 'MarkerFaceColor', 'b' );
%blue diamond
 plot( goalX, goalY, 'bp','MarkerSize', 14, 'MarkerFaceColor', 'b' ); %blue
pentagram
 grid on;
 axis( [0 dim 0 dim] );
 set( gca, 'YTick', 0:1:dim );
 set( gca, 'XTick', 0:1:dim );
 set( gca, 'GridLineStyle', '-' );
if( save_maze )
 filename = sprintf( '%s/image_%d.ppm', img_dir, i );
 fprintf( 'saving %s...', filename );
 print( map, '-dppm', '-r200', filename );
 fprintf( 'done\n' );
 close( map );
else
 drawnow;
end
Qsum = sum( Q, 3 );
if( save_maze )
 img = figure( 'Visible', 'off' );
 surf( Qsum );
 filename = sprintf( '%s/Qsum.ppm', img_dir );
 fprintf( 'saving %s...', filename );
 print( img, '-dppm', '-r200', filename );
 fprintf( 'done\n' );
else
 figure;
 surf( Qsum );
end