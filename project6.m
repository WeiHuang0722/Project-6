%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Wei Huang, UC Berkeley Materials Science and Engineering, wei_huang@berkeley.edu
% Written for MSE C286, Project 6, Spring 2020.
% ID: 3035350007

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
clear all; close all; clc;
%% CONSTANTS
W1 = 1/3; W2 = 1/3; W3 = 1/3; % cost function weights
S = 100; % number of strings per generation
parents = 10; 
children = parents; % breeding population parameters
G = 5000; % total generations

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% DEFINE ALL OTHER REQUIRED VALUES
% DEFINE PHYSICAL CONSTANTS
k_1 = 80E9; % phase 1 bulk modulus, Pa
mu_1 = 30E9; % phase 1 shear modulus, Pa
s_1 = 1.0E7; % phase 1 electrical conductivity, S/m
K_1 = 4.3; % phase 1 thermal conductivity, W/m-K

% INITIALIZE STORAGE ARRAYS (PI, etc)
comp_time = zeros(G, S); % array to store computation time for each design string
PI = ones(G, S); % store overall performance for all strings
Orig = zeros(G, S); % place to store history of where best design in each generation came from

Dev = ones(1, S); % relative deviation square
Pi = ones(1, S); % cost function array 

g = 0; % initialize generation
L = rand(S , 5); % design strings

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% GENETIC ALGORITHM (Keeping Top 10 Parents after Each Genereation)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while g <= G  % FOR EACH GENERATION
    g = g + 1
    for s = 1:S % FOR EACH STRING
        tic 
        % define J and d_o by taking them from the design string
        if (g == 1 || s > parents + children)
        L(s , 1) = 9*k_1*rand + k_1; k_2 = L(s , 1); 
        L(s , 2) = 9*mu_1*rand + mu_1; mu_2 = L(s , 2);
        L(s , 3) = 9*s_1*rand + s_1; s_2 = L(s , 3); 
        L(s , 4) = 9*K_1*rand + K_1; K_2 = L(s , 4); 
        L(s , 5) = 2/3*rand; v_2 = L(s , 5); v_1 = 1 - v_2;
        else
        k_2 = L(s , 1); mu_2 = L(s , 2); s_2 = L(s , 3); 
        K_2 = L(s , 4); v_2 = L(s , 5); v_1 = 1 - v_2;
        end
        % RUN SIMULATION
        [Pi_electrical, Pi_thermal, Pi_mechanical] = mat_opt(k_2, mu_2, s_2, K_2, v_2, v_1); 
        
        % COMPUTE COST FUNCTION
        Pi(s) = W1*Pi_electrical + W2*Pi_thermal + W3*Pi_mechanical;
    myProgressBar(sum(comp_time(:)), s + S*(g - 1), S*G)
    comp_time(g, s) = toc;
    end   % END STRING
    
    % SORT
    [Pi, I] = sort(Pi); % rank the strings
    Orig(g, :) = I; % save sorting order for family tree
    L = L(I, :); % sort design strings
    PI(g, :) = Pi; % save sorted costs

    % BREED
    breeders = L(1:parents, :); % make a copy of the top performers
    for j =  1:2:children % for each pair
        phi1 = rand(1, 5);
        phi2 = rand(1, 5);
        L(parents + j, :) = phi1 .*breeders(j, :) + (1 - phi1) .* breeders(j + 1, :);
        L(parents + j + 1, :) = phi2 .*breeders(j, :) + (1 - phi2) .* breeders(j + 1, :);
    end
    
    if mod(g, G/10) == 0 % plot results every few generations and at end
        figure(1); clf; familyTree(Orig(1:g, :), parents, parents); 
    end
end% END GENERATIONS

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %% GENETIC ALGORITHM (Not Keeping Top 10 Parents after Each Generation)
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% while g <= G  % FOR EACH GENERATION
%     g = g + 1
%     for s = 1:S % FOR EACH STRING
%         tic 
%         % define J and d_o by taking them from the design string
%         if (g == 1 || s > children)
%         L(s , 1) = 9*k_1*rand + k_1; k_2 = L(s , 1); 
%         L(s , 2) = 9*mu_1*rand + mu_1; mu_2 = L(s , 2);
%         L(s , 3) = 9*s_1*rand + s_1; s_2 = L(s , 3); 
%         L(s , 4) = 9*K_1*rand + K_1; K_2 = L(s , 4); 
%         L(s , 5) = 2/3*rand; v_2 = L(s , 5); v_1 = 1 - v_2;
%         else
%         k_2 = L(s , 1); mu_2 = L(s , 2); s_2 = L(s , 3); 
%         K_2 = L(s , 4); v_2 = L(s , 5); v_1 = 1 - v_2;
%         end
%         % RUN SIMULATION
%         [Pi_electrical, Pi_thermal, Pi_mechanical] = mat_opt(k_2, mu_2, s_2, K_2, v_2, v_1); 
%         
%         % COMPUTE COST FUNCTION
%         Pi(s) = W1*Pi_electrical + W2*Pi_thermal + W3*Pi_mechanical;
%     myProgressBar(sum(comp_time(:)), s + S*(g - 1), S*G)
%     comp_time(g, s) = toc;
%     end   % END STRING
%     
%     % SORT
%     [Pi, I] = sort(Pi); % rank the strings
%     Orig(g, :) = I; % save sorting order for family tree
%     L = L(I, :); % sort design strings
%     PI(g, :) = Pi; % save sorted costs
%     % CHECK IF TOL SATISFIED. if it is, break
%   %  if min(Pi) <= TOL
%   %     break
%   %  end
%     % BREED
%     breeders = L(1:parents, :); % make a copy of the top performers
%     for j =  1:2:children % for each pair
%         phi1 = rand(1, 5);
%         phi2 = rand(1, 5);
%         L(parents, :) = phi1 .*breeders(j, :) + (1 - phi1) .* breeders(j + 1, :);
%         L(parents + 1, :) = phi2 .*breeders(j, :) + (1 - phi2) .* breeders(j + 1, :);
%     end
%     
%     if mod(g, G/10) == 0 % plot results every few generations and at end
%         figure(1); clf; familyTree(Orig(1:g, :), parents, parents); 
%     end
% end% END GENERATIONS
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %% GENETIC ALGORITHM (Keeping Top 10, Weights: -.5 <= phi <= 1.5)
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% while g <= G  % FOR EACH GENERATION
%     g = g + 1
%     for s = 1:S % FOR EACH STRING
%         tic 
%         % define J and d_o by taking them from the design string
%         if (g == 1 || s > parents + children)
%         L(s , 1) = 9*k_1*rand + k_1; k_2 = L(s , 1); 
%         L(s , 2) = 9*mu_1*rand + mu_1; mu_2 = L(s , 2);
%         L(s , 3) = 9*s_1*rand + s_1; s_2 = L(s , 3); 
%         L(s , 4) = 9*K_1*rand + K_1; K_2 = L(s , 4); 
%         L(s , 5) = 2/3*rand; v_2 = L(s , 5); v_1 = 1 - v_2;
%         else
%         k_2 = L(s , 1); mu_2 = L(s , 2); s_2 = L(s , 3); 
%         K_2 = L(s , 4); v_2 = L(s , 5); v_1 = 1 - v_2;
%         end
%         % RUN SIMULATION
%         [Pi_electrical, Pi_thermal, Pi_mechanical] = mat_opt(k_2, mu_2, s_2, K_2, v_2, v_1); 
%         
%         % COMPUTE COST FUNCTION
%         Pi(s) = W1*Pi_electrical + W2*Pi_thermal + W3*Pi_mechanical;
%     myProgressBar(sum(comp_time(:)), s + S*(g - 1), S*G)
%     comp_time(g, s) = toc;
%     end   % END STRING
%     
%     % SORT
%     [Pi, I] = sort(Pi); % rank the strings
%     Orig(g, :) = I; % save sorting order for family tree
%     L = L(I, :); % sort design strings
%     PI(g, :) = Pi; % save sorted costs
%     % CHECK IF TOL SATISFIED. if it is, break
%   %  if min(Pi) <= TOL
%   %     break
%   %  end
%     % BREED
%     breeders = L(1:parents, :); % make a copy of the top performers
%     for j =  1:2:children % for each pair
%         phi1 = 2*rand(1, 5) - .5;
%         phi2 = 2*rand(1, 5) - .5;
%         L(parents + j, :) = phi1 .*breeders(j, :) + (1 - phi1) .* breeders(j + 1, :);
%         L(parents + j + 1, :) = phi2 .*breeders(j, :) + (1 - phi2) .* breeders(j + 1, :);
%     end
%     
%     if mod(g, G/10) == 0 % plot results every few generations and at end
%         figure(1); clf; familyTree(Orig(1:g, :), parents, parents); 
%     end
% end% END GENERATIONS
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% PRESENT RESULTS

figure()
plotCost(PI, parents, g)
figure()
plot(1:g, sum(comp_time(1:g, :), 2), '-', 'Color', [.8 .2 .6], 'LineWidth', 1);
ylabel("Runtime (s)"); xlabel("Generation"); title("Runtime History");

fprintf("Total run time: " + sec2Clock(sum(comp_time(:))) + "\n\n");  
%% HELPER FUNCTIONS
function plotCost(PI, parents, g)

% Plot information about the average cost for subsets of the population at
% each generation. Assumes that PI is a G by S array, where each position
% represents the cost of a given design string in a given generation after
% sorting. Should produce a plot like the one in the assignment handout. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% a general method for plotting the cost convergence of a series of genetic
% algorithm (or similar) iuterative design options.
loglog(1:g, PI(1:g, 1), '-', 'LineWidth', 3); hold on;
loglog(1:g, mean(PI(1:g, 1:parents), 2), 'LineStyle', '-', 'Color', [.6 .7 .1], 'LineWidth', 3); hold on;
loglog(1:g, mean(PI(1:g, :), 2), '--', 'LineWidth', 3);
title("Cost"); xlabel("Generation"); ylabel("\Pi");
legend("Best", "Parent Mean", "Overall Mean", 'location', 'best'); shg

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
%% Material Optimization.m
function [Pi_electrical, Pi_thermal, Pi_mechanical] = mat_opt(k_2, mu_2, s_2, K_2, v_2, v_1)
%% Define Constants
w1 = 1; wj = 0.5; 

phi = 0.5; % Hashin Shtrikman combination constant
TOL_k = 0.5; % bulk modulus tolerance
TOL_mu = 0.5; % shear modulus tolerance
TOL_K = 0.5; % thermal tolerance
TOL_s = 0.8; % electrical tolerance
k_des = 111E9; % desired effective bulk modulus, Pa
mu_des = 47E9; % desired effective shear modulus, Pa
s_des = 2.0E7; % desired effective electrical conductivity, S/m
K_des = 6.2; % desired effective thermal conductivity, W/m-K
% DEFINE PHYSICAL CONSTANTS
k_1 = 80E9; % phase 1 bulk modulus, Pa
mu_1 = 30E9; % phase 1 shear modulus?Pa
s_1 = 1.0E7; % phase 1 electrical conductivity, S/m
K_1 = 4.3; % phase 1 thermal conductivity, W/m-K
%% Initial Variables
k_ast_lo = k_1 + v_2/(1/(k_2 - k_1) + 3*(1 - v_2)/(3*k_1 + 4*mu_1)); % lower bound of effective bulk modulus
k_ast_up = k_2 + (1 - v_2)/(1/(k_1 - k_2) + 3*v_2/(3*k_2 + 4*mu_2)); % upper bound of effective bulk modulus
mu_ast_lo = mu_1 + v_2/(1/(mu_2 - mu_1) + 6*(1 - v_2)*(k_1 + 2*mu_1)/(5*mu_1*(3*k_1+4*mu_1))); % lower bound of effective shear modulus
mu_ast_up = mu_2 + (1 - v_2)/(1/(mu_1 - mu_2) + 6*v_2*(k_2 + 2*mu_2)/(5*mu_2*(3*k_2+4*mu_2))); % upper bound of effective shear modulus

s_ast_lo = s_1 + v_2/(1/(s_2 - s_1) + (1 - v_2)/3*s_1); % lower bound of effective electrical conductivity
s_ast_up = s_2 + (1 - v_2)/(1/(s_1 - s_2) + v_2/3*s_2); % upper bound of effective electrical conductivity

K_ast_lo = K_1 + v_2/(1/(K_2 - K_1) + (1 - v_2)/3*K_1); % lower bound of effective thermal conductivity
K_ast_up = K_2 + (1 - v_2)/(1/(K_1 - K_2) + v_2/3*K_2); % upper bound of effective thermal conductivity
%% Mechanical Properties
k_ast = phi*k_ast_up + (1 - phi)*k_ast_lo; % effective bulk modulus
mu_ast = phi*mu_ast_up + (1 - phi)*mu_ast_lo; % effective shear modulus
C_k_2 = (1/v_2)*(k_2/k_ast)*((k_ast - k_1)/(k_2 - k_1)); % phase 2 bulk modulus concentration factor
C_k_1 = (1/v_1)*(1 - v_2*C_k_2); % phase 1 bulk modulus concentration factor
C_mu_2 = (1/v_2)*(mu_2/mu_ast)*((mu_ast - mu_1)/(mu_2 - mu_1)); % phase 2 shear modulus concentration factor
C_mu_1 = (1/v_1)*(1 - v_2*C_mu_2); % phase 1 shear modulus concentration factor

Pi_mechanical = w1*abs((k_des - k_ast)/k_des) + wj*logical((mu_des - mu_ast)/mu_des > 0)*abs((mu_des - mu_ast)/mu_des)... 
+ wj*logical((C_k_2 - TOL_k)/TOL_k > 0)*abs((C_k_2 - TOL_k)/TOL_k) + wj*logical((C_mu_2 - TOL_mu)/TOL_mu > 0)*abs((C_mu_2 - TOL_mu)/TOL_mu)...
+ wj*logical((C_k_1 - TOL_k)/TOL_k > 0)*abs((C_k_1 - TOL_k)/TOL_k) + wj*logical((C_mu_1 - TOL_mu)/TOL_mu > 0)*abs((C_mu_1 - TOL_mu)/TOL_mu);
%% Electrical Properties
s_ast = phi*s_ast_up + (1 - phi)*s_ast_lo; % effective electrical conductivity
C_E1_C_J1 = (s_1/s_ast)*((1/(1 - v_2))*((s_2 - s_ast)/(s_2 - s_1)))^2; % phase 1 electrical concentration factor
C_E2_C_J2 = (s_2/s_ast)*((1/v_2)*((s_ast - s_1)/(s_2 - s_1)))^2; % phase 2 electrical concentration factor

Pi_electrical = w1*abs((s_des - s_ast)/s_des) + wj*logical((C_E1_C_J1 - TOL_s)/TOL_s > 0)*abs((C_E1_C_J1 - TOL_s)/TOL_s)...
+ wj*logical((C_E2_C_J2 - TOL_s)/TOL_s > 0)*abs((C_E2_C_J2 - TOL_s)/TOL_s);
%% Thermal Properties
K_ast = phi*K_ast_up + (1 - phi)*K_ast_lo; % effective thermal conductivity
C_q_2 = (K_2/K_ast)*(1/v_2)*((K_ast - K_1)/(K_2 - K_1)); % phase 2 thermal concentration factor
C_q_1 = (K_1/K_ast)*(1/(1 - v_2))*((K_2 - K_ast)/(K_2 - K_1)); % phase 1 thermal concentration factor

Pi_thermal = w1*abs((K_des - K_ast)/K_des) + wj*logical((C_q_1 - TOL_K)/TOL_K > 0)*abs((C_q_1 - TOL_K)/TOL_K)...
+ wj*logical((C_q_2 - TOL_K)/TOL_K > 0)*abs((C_q_2 - TOL_K)/TOL_K);
end

% familyTree.m 
function familyTree(Orig, parents, pop)
%%

% Inputs:
%   Orig -- the indices of a sorted GA generation from before sorting (see sort() documentation),
%   parents -- the number of parents, required for interpreting source
%   pop --  the number of performers to plot. Use pop >= parents.

% Returns:
% no variables, plots a representation of the evolution history to the current figure.

% The function automatically ignores generations where no rank changes
% occur among the parents OR there are any repeated indices (indicating
% incorrect data).

% Data visualization: This function can be used to visualize and interpret the performance of
% your GA iteration process. Gray lines represent "survival" with or
% without a rank change. Red lines represent breeding (e.g.,  a new string
% will be connected to its parents with red lines). New random strings have
% no connections. A surviving string will be represented with gray, a new
% string generated randomly will be a blue mark and a new string generated
% by breeding will be red.

% Performance interpretation: If your GA is working correctly, there should
% be lots of turnover (i.e., few generations where nothing happens),
% significant numbers of successful offspring and a moderate number of
% successful random strings. You can also spot stagnation (a parent
% surviving for many generations and continually producing offspring that
% stay in the top ranks).

%%

Orig2 = Orig(:, 1:pop); % trim source to just relevant entries.
row = 0; changes = []; % initialize variables for determining relevant generations
children = parents; % assume nearest-neighbor with two children
rando = zeros(0, 2); inc = zeros(0, 2); kid = zeros(0, 2); % intialize empty storage arrays
G = size(Orig, 1); % total number of generations.
pts = 25; % number of points to plot in connections
c1 = [.6 .6 .6]; c2 = [1 .6 .6]; % line colors for surviving connections and children
lw = 1.5; % connection line weight
mw = 1.5; % marker line weight

incx = zeros(pts, 2); incy = zeros(pts, 2); % empty arrays for connecting line coordinates.
kidx = zeros(pts, 2); kidy = zeros(pts, 2);

for g = 1:G % for every generation
    if ~isequal(Orig2(g, 1:parents), 1:parents) && length(unique(Orig2(g, :))) == pop % if a change in survivors and valid data
        row = row + 1; % row on which to plot current state - counts relevant generations
        x1 = row - 1; x2 = row; % start and end points of connections
        changes = [changes; g]; % record that a change occured in this generation
        for i = 1:pop
            s = Orig2(g, i); y2 = i;
            if s == i && i <= parents && g > 1 % if the entry is a surviving parent who has not moved
                y1 = i;
                [xx, yy] = mySpline([x1 x2], [y1 y2], pts);
                incx = [incx, xx]; incy = [incy, yy];
                inc = [inc; [x2, y2]];
            elseif  s <= parents && g > 1% if the entry is a surviving parent who has been moved down
                y1 = s;
                [xx, yy] = mySpline([x1 x2], [y1 y2], pts);
                incx = [incx, xx]; incy = [incy, yy];
                inc = [inc; [x2, y2]];
            elseif s <= parents + children && g > 1 % if the entry is a child
                for n = 2:2:children
                    if s <= parents + n
                        y11 = n - 1; y12 = n;
                        [xx1, yy1] = mySpline([x1, x2], [y11, y2], pts);
                        [xx2, yy2] = mySpline([x1, x2], [y12, y2], pts);
                        kidx = [kidx, xx1, xx2]; kidy = [kidy, yy1, yy2];
                        kid = [kid; [x2, y2]];
                        break
                    end
                end
            else % if it's a new random addition.
                rando = [rando; [x2, y2]];
            end
        end
    end
end

p1 = plot(incx, incy, '-', 'Color', c1, 'LineWidth', 1.5); hold on
p2 = plot(kidx, kidy, '-', 'Color', c2, 'LineWidth', 1.5); hold on
p3 = plot(rando(:, 1), rando(:, 2), 's', 'MarkerEdgeColor', [.2 .4 .9], 'MarkerFaceColor', [.6 .6 1], 'MarkerSize', 10, 'LineWidth', mw); hold on % plot random
p4 = plot(inc(:, 1), inc(:, 2), 's', 'MarkerEdgeColor', [.3 .3 .3], 'MarkerFaceColor', [.6 .6 .6], 'MarkerSize', 10, 'LineWidth', mw); hold on % plot survival
p5 = plot(kid(:, 1), kid(:, 2), 's', 'MarkerEdgeColor', [.9 .3 .3], 'MarkerFaceColor', [1 .6 .6], 'MarkerSize', 10, 'LineWidth', mw); % plot children
h = [p3, p4, p5];
legend(h, "Random", "Incumbent", "Child")

xlabels = {};
for i = 1:numel(changes)
    xlabels{i} = num2str(changes(i));
end

ylabels = {};
for j = 1:pop
    ylabels{j} = num2str(j);
end

title("Family Tree");
set(gca, 'xtick', [1:row]); set(gca,'ytick', [1:pop]);
set(gca,'xticklabel', xlabels); set(gca,'yticklabel', ylabels);
xlabel("generation"); ylabel("Rank");
axis([0 row + 1 0 pop + 1]); view([90, 90])
end

function [xx, yy] = mySpline(x, y, pts)
% produces clamped splines between two points

% Inputs:
%   x -- 2-value vector containing the x coordinates of the end points
%   y -- 2-value vector containing the y coordinates of the end points
%   pts -- the number of total points to plot (ends plus intermediates)

% Returns:
%    xx -- array of x coordinates to plot
%    yy -- array of y coordinates to plot

cs = spline(x, [0 y 0]);
xx = linspace(x(1), x(2), pts)';
yy = ppval(cs,xx);
end

% myProgressBar.m
function myProgressBar(tock, t, tf, m)
% Provides a status update on a calculation, printed to the command window.
% avoids the issues associated with the GUI-based equilvalent functions.
%
% Inputs:
%   tock -- total run time so far, seconds
%   t -- cycles, simulation time, etc completed
%   tf -- total cycles, etc required for process
%   m -- optional message input, empty by default

% Outputs:
% no variables, prints information to the command window in the format:
% "#######-----------" + "m" + "Time remaining: " + "HH:MM:SS"

if nargin < 4
    m = '';
end
rem = max(tock*(tf/t - 1), 0);
clock = sec2Clock(rem);
totchar = 20;
fprintf(repeatStr("#", floor(t/tf*totchar)) + repeatStr("-", totchar - floor(t/tf*totchar)) ...
    + "   " + m + "   Time remaining: " + clock + "\n\n");
end

function out = sec2Clock(s)
% returns a string of characters that represents a number of seconds in
% format HH:MM:SS. Can include arbitrarily many hour digits to account for
% large times. Rounded down to nearest number of seconds.
remh = floor(s / 3600); s = s - 3600*remh; remm = floor(s / 60); s = s - 60*remm; rems = floor(s);
out = padStr(num2str(remh), "0", 2)  + ":" + padStr(num2str(remm), "0", 2) ...
    + ":" + padStr(num2str(rems), "0", 2);
end

function out = padStr(st, pad, minlength)
% returns string st plus some number of pad characters such that the total
% string length is as least minlength. Puts the padding on the left side.
out = st;
while (strlength(out) < minlength) out = pad + out; end
end

function out = repeatStr(st, n)
% returns a given string st repeated n times
out = ""; for i = 1:n out = out + st; end
end
