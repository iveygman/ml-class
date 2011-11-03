function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% each node in hidden layer...
a1 = [ones(m,1) X];
z2 = zeros(size(Theta1,1),m);
a2 = z2;
for k = 1:size(Theta1,1)
	z2(k,:) = (Theta1(k,:)*a1')';
end
a2 = sigmoid(z2');
a2 = [ones(size(a2,1),1) a2];

% each node in output layer...
z3 = zeros(num_labels,m);
for k = 1:num_labels
	z3(k,:) = Theta2(k,:)*a2';
end

size(sigmoid(z3))

[dummy p] = max(sigmoid(z3)',[],2);

% =========================================================================


end
