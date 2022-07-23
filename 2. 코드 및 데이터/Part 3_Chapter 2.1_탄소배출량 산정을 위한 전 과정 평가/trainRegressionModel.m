function [trainedModel, validationRMSE] = trainRegressionModel(trainingData)
% [trainedModel, validationRMSE] = trainRegressionModel(trainingData)
% 훈련된 회귀 모델과 그 RMSE을(를) 반환합니다. 이 코드는 회귀 학습기 앱에서 훈련된 모델
% 을 다시 만듭니다. 생성된 코드를 사용하여 동일한 모델을 새 데이터로 훈련시키는 것을 자동
% 화하거나, 모델을 프로그래밍 방식으로 훈련시키는 방법을 익힐 수 있습니다.
%
%  입력값:
%      trainingData: 앱으로 가져온 것과 동일한 예측 변수와 응답 변수 열을 포함하는 테
%       이블입니다.
%
%  출력값:
%      trainedModel: 훈련된 회귀 모델이 포함된 구조체입니다. 이 구조체에는 훈련된 모
%       델에 대한 정보가 포함된 다양한 필드가 들어 있습니다.
%
%      trainedModel.predictFcn: 새 데이터를 사용하여 예측하기 위한 함수입니다.
%
%      validationRMSE: RMSE를 포함하는 double형입니다. RMSE는 앱의 내역 목록에 각
%       모델별로 표시됩니다.
%
% 새 데이터로 모델을 훈련시키려면 이 코드를 사용하십시오. 모델을 다시 훈련시키려면 명령줄
% 에서 원래 데이터나 새 데이터를 입력 인수 trainingData(으)로 사용하여 함수를 호출하십
% 시오.
%
% 예를 들어, 원래 데이터 세트 T(으)로 훈련된 회귀 모델을 다시 훈련시키려면 다음을 입력하
% 십시오.
%   [trainedModel, validationRMSE] = trainRegressionModel(T)
%
% 새 데이터 T2에서 반환된 'trainedModel'을(를) 사용하여 예측하려면 다음을 사용하십시
% 오.
%   yfit = trainedModel.predictFcn(T2)
%
% T2은(는) 적어도 훈련 중에 사용된 것과 동일한 예측 변수 열을 포함하는 테이블이어야 합니
% 다. 세부 정보를 보려면 다음을 입력하십시오.
%   trainedModel.HowToPredict

% MATLAB에서 2022-02-08 22:31:50에 자동 생성됨


% 예측 변수와 응답 변수 추출
% 이 코드는 모델을 훈련시키기에 적합한 형태로 데이터를
% 처리합니다.

trainingData = readtable('paperdata.xlsx','Range', 'A1:E201')
inputTable = trainingData;
predictorNames = {'stockFlowLmin', 'talcFlowLmin', 'pressurekgcm2', 'speedmmin'};
predictors = inputTable(:, predictorNames);
response = inputTable.emission;
isCategoricalPredictor = [false, false, false, false];

% 회귀 모델 훈련
% 이 코드는 모든 모델 옵션을 지정하고 모델을 훈련시킵니다.
responseScale = iqr(response);
if ~isfinite(responseScale) || responseScale == 0.0
    responseScale = 1.0;
end
boxConstraint = responseScale/1.349;
epsilon = responseScale/13.49;
regressionSVM = fitrsvm(...
    predictors, ...
    response, ...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', boxConstraint, ...
    'Epsilon', epsilon, ...
    'Standardize', true);

% 예측 함수를 사용하여 결과 구조체 생성
predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(regressionSVM, x);
trainedModel.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% 추가적인 필드를 결과 구조체에 추가
trainedModel.RequiredVariables = {'pressurekgcm2', 'speedmmin', 'stockFlowLmin', 'talcFlowLmin'};
trainedModel.RegressionSVM = regressionSVM;
trainedModel.About = '이 구조체는 회귀 학습기 R2021a에서 내보낸 훈련된 모델입니다.';
trainedModel.HowToPredict = sprintf('새 테이블 T를 사용하여 예측하려면 다음을 사용하십시오. \n yfit = c.predictFcn(T) \n여기서 ''c''를 이 구조체를 나타내는 변수의 이름(예: ''trainedModel'')으로 바꾸십시오. \n \n테이블 T는 다음에서 반환된 변수를 포함해야 합니다. \n c.RequiredVariables \n변수 형식(예: 행렬/벡터, 데이터형)은 원래 훈련 데이터와 일치해야 합니다. \n추가 변수는 무시됩니다. \n \n자세한 내용은 <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appregression_exportmodeltoworkspace'')">How to predict using an exported model</a>을(를) 참조하십시오.');

% 예측 변수와 응답 변수 추출
% 이 코드는 모델을 훈련시키기에 적합한 형태로 데이터를
% 처리합니다.
inputTable = trainingData;
predictorNames = {'stockFlowLmin', 'talcFlowLmin', 'pressurekgcm2', 'speedmmin'};
predictors = inputTable(:, predictorNames);
response = inputTable.emission;
isCategoricalPredictor = [false, false, false, false];

% 홀드아웃 검증 설정
cvp = cvpartition(size(response, 1), 'Holdout', 0.4);
trainingPredictors = predictors(cvp.training, :);
trainingResponse = response(cvp.training, :);
trainingIsCategoricalPredictor = isCategoricalPredictor;

% 회귀 모델 훈련
% 이 코드는 모든 모델 옵션을 지정하고 모델을 훈련시킵니다.
responseScale = iqr(trainingResponse);
if ~isfinite(responseScale) || responseScale == 0.0
    responseScale = 1.0;
end
boxConstraint = responseScale/1.349;
epsilon = responseScale/13.49;
regressionSVM = fitrsvm(...
    trainingPredictors, ...
    trainingResponse, ...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', boxConstraint, ...
    'Epsilon', epsilon, ...
    'Standardize', true);

% 예측 함수를 사용하여 결과 구조체 생성
svmPredictFcn = @(x) predict(regressionSVM, x);
validationPredictFcn = @(x) svmPredictFcn(x);

% 추가적인 필드를 결과 구조체에 추가


% 검증 예측값 계산
validationPredictors = predictors(cvp.test, :);
validationResponse = response(cvp.test, :);
validationPredictions = validationPredictFcn(validationPredictors);

% 검증 RMSE 계산
isNotMissing = ~isnan(validationPredictions) & ~isnan(validationResponse);
validationRMSE = sqrt(nansum(( validationPredictions - validationResponse ).^2) / numel(validationResponse(isNotMissing) ));

% 생성된 모델을 이용한 이산화탄소 발생량 예측
T = readtable('newdata.xlsx','Range','A1:D2')
yfit=trainedModel.predictFcn(T)
