#!/bin/bash

# KBSME 자살 예측 모델 - 마스터 실험 실행 스크립트
# 전체 5단계 실험을 순차적으로 실행하거나 개별 실행 가능

# 실험 중단 처리 함수
cleanup_on_exit() {
    echo ""
    echo "⚠️  실험이 중단되었습니다."
    echo "현재 진행 중인 프로세스를 정리합니다..."
    pkill -f "run_hyperparameter_tuning"
    pkill -f "run_resampling_experiment"
    echo "정리 완료."
    exit 1
}

# 시그널 핸들러 설정
trap cleanup_on_exit INT TERM

# 명령어 인자 검증 함수
validate_commands() {
    echo "🔍 명령어 인자 검증 중..."
    
    # 필수 스크립트 존재 확인
    local scripts=("phase1_baseline.sh" "phase2_input_range.sh" "phase3_resampling.sh" "phase4_deep_analysis.sh" "phase5_final_optimization.sh")
    local missing_scripts=()
    
    for script in "${scripts[@]}"; do
        if [[ ! -f "$script" ]]; then
            missing_scripts+=("$script")
        fi
    done
    
    if [[ ${#missing_scripts[@]} -gt 0 ]]; then
        echo "❌ 다음 스크립트 파일들이 없습니다:"
        for script in "${missing_scripts[@]}"; do
            echo "   - $script"
        done
        echo ""
        echo "먼저 모든 Phase 스크립트를 생성해주세요."
        return 1
    fi
    
    # Python 스크립트 존재 확인
    if [[ ! -f "scripts/run_hyperparameter_tuning.py" ]]; then
        echo "❌ scripts/run_hyperparameter_tuning.py 파일이 없습니다."
        return 1
    fi
    
    if [[ ! -f "scripts/run_resampling_experiment.py" ]]; then
        echo "❌ scripts/run_resampling_experiment.py 파일이 없습니다."
        return 1
    fi
    
    # 데이터 파일 확인
    if [[ ! -f "data/sourcedata/data.csv" ]]; then
        echo "⚠️  data/sourcedata/data.csv 파일을 확인할 수 없습니다."
        echo "데이터 파일 경로를 확인해주세요."
        return 1
    fi
    
    echo "✅ 모든 필수 파일 확인 완료"
    return 0
}

echo "=================================================================="
echo "🧠 KBSME 자살 예측 모델 - 종합 실험 시스템"
echo "=================================================================="
echo "Phase 1: 기준선 설정 (4개 모델 기본 성능)"
echo "Phase 2: Input 범위 조정 (데이터 크기, 피처 선택)"
echo "Phase 3: Resampling 비교 (35개 리샘플링 실험)"
echo "Phase 4: 모델 심층 분석 (고성능 튜닝, 특성 분석)"
echo "Phase 5: 통합 최적화 (최종 모델 구축)"
echo ""
echo "총 예상 실험 수: 77개"
echo "총 예상 소요 시간: 2-4일 (시스템 사양에 따라)"
echo "=================================================================="

# 명령어 인자 검증
if ! validate_commands; then
    echo "❌ 검증 실패. 실험을 중단합니다."
    exit 1
fi

# 환경 확인
echo "🔍 실험 환경 확인 중..."
echo "현재 디렉토리: $(pwd)"
echo "Python 환경: $(which python)"
echo "MLflow 설치: $(pip list | grep mlflow | head -1 || echo '❌ MLflow 미설치')"

# 시스템 리소스 확인
echo ""
echo "💻 시스템 리소스 확인 중..."
python scripts/check_cpu_usage.py

echo ""

# 실행 모드 선택
echo "실행 모드를 선택해주세요:"
echo "1) 전체 실험 순차 실행 (Phase 1-5)"
echo "2) 개별 Phase 선택 실행"
echo "3) 빠른 테스트 모드 (소규모 데이터)"
echo "4) 환경 설정 및 초기 분석만"
echo "5) 실험 상태 확인 및 MLflow UI"
echo "6) 실험 중단 및 정리"
echo ""
read -p "선택 (1-6): " -n 1 -r mode
echo ""

case $mode in
    1)
        echo "🚀 전체 실험 순차 실행을 시작합니다."
        echo ""
        
        # 전체 실험 시작 시간 기록
        overall_start=$(date)
        echo "전체 실험 시작 시간: $overall_start"
        
        # 사용자 확인
        echo "⚠️  주의사항:"
        echo "- 전체 실험은 2-4일이 소요될 수 있습니다"
        echo "- 시스템 리소스를 많이 사용합니다"
        echo "- 중간에 중단하면 해당 Phase부터 재시작 가능합니다"
        echo "- Ctrl+C로 언제든지 안전하게 중단할 수 있습니다"
        echo ""
        read -p "계속 진행하시겠습니까? (y/n): " -n 1 -r confirm
        echo ""
        
        if [[ $confirm =~ ^[Yy]$ ]]; then
            # Phase별 순차 실행
            for i in {1..5}; do
                script="phase${i}_"
                case $i in
                    1) script+="baseline.sh"; phase_name="기준선 설정" ;;
                    2) script+="input_range.sh"; phase_name="Input 범위 조정" ;;
                    3) script+="resampling.sh"; phase_name="Resampling 비교" ;;
                    4) script+="deep_analysis.sh"; phase_name="모델 심층 분석" ;;
                    5) script+="final_optimization.sh"; phase_name="통합 최적화" ;;
                esac
                
                echo ""
                echo "========================================"
                echo "🔄 Phase $i ($phase_name) 실행 중..."
                echo "========================================"
                
                chmod +x "$script"
                if ./"$script"; then
                    echo "✅ Phase $i 완료!"
                    
                    # Phase 간 중간 결과 확인 옵션
                    if [[ $i -lt 5 ]]; then
                        echo ""
                        read -p "Phase $i 결과를 MLflow UI에서 확인하시겠습니까? (y/n): " -n 1 -r check_result
                        echo
                        if [[ $check_result =~ ^[Yy]$ ]]; then
                            echo "MLflow UI 실행 중... (http://localhost:5000)"
                            mlflow ui --host 0.0.0.0 --port 5000 &
                            echo "백그라운드에서 MLflow UI가 실행되었습니다."
                            read -p "결과 확인 후 Enter를 눌러 다음 Phase로 진행하세요..."
                        fi
                    fi
                else
                    echo "❌ Phase $i 실행 중 오류 발생. 로그를 확인해주세요."
                    read -p "계속 진행하시겠습니까? (y/n): " -n 1 -r continue_confirm
                    echo ""
                    if [[ ! $continue_confirm =~ ^[Yy]$ ]]; then
                        echo "실험을 중단합니다."
                        cleanup_on_exit
                    fi
                fi
                
                # Phase 간 대기 시간 (선택사항)
                if [[ $i -lt 5 ]]; then
                    echo ""
                    echo "다음 Phase로 넘어가기 전 10초 대기..."
                    sleep 10
                fi
            done
            
            overall_end=$(date)
            echo ""
            echo "🎉 전체 실험이 완료되었습니다!"
            echo "시작: $overall_start"
            echo "종료: $overall_end"
            echo ""
            echo "📊 결과 확인을 위해 MLflow UI를 실행합니다."
            mlflow ui --host 0.0.0.0 --port 5000 &
        else
            echo "실험을 취소했습니다."
        fi
        ;;
        
    2)
        echo "개별 Phase 선택 실행"
        echo ""
        echo "실행할 Phase를 선택해주세요:"
        echo "1) Phase 1: 기준선 설정"
        echo "2) Phase 2: Input 범위 조정"
        echo "3) Phase 3: Resampling 비교"
        echo "4) Phase 4: 모델 심층 분석"
        echo "5) Phase 5: 통합 최적화"
        echo ""
        read -p "Phase 번호 (1-5): " -n 1 -r phase_num
        echo ""
        
        case $phase_num in
            1) script="phase1_baseline.sh"; phase_name="기준선 설정" ;;
            2) script="phase2_input_range.sh"; phase_name="Input 범위 조정" ;;
            3) script="phase3_resampling.sh"; phase_name="Resampling 비교" ;;
            4) script="phase4_deep_analysis.sh"; phase_name="모델 심층 분석" ;;
            5) script="phase5_final_optimization.sh"; phase_name="통합 최적화" ;;
            *) echo "잘못된 선택입니다."; exit 1 ;;
        esac
        
        echo "🚀 Phase $phase_num ($phase_name) 실행을 시작합니다."
        chmod +x "$script"
        ./"$script"
        ;;
        
    3)
        echo "🏃‍♂️ 빠른 테스트 모드"
        echo ""
        echo "소규모 데이터로 빠른 테스트를 진행합니다."
        echo "- 데이터: 1,000행"
        echo "- n_trials: 10"
        echo "- 각 모델 1회씩만 실행"
        echo ""
        
        echo ">>> XGBoost 빠른 테스트"
        python scripts/run_hyperparameter_tuning.py \
            --model-type xgboost \
            --experiment-type hyperparameter_tuning \
            --nrows 1000 \
            --n-trials 10 \
            --experiment-name "quick_test_xgboost" \
            --verbose 2
        
        if [ $? -ne 0 ]; then
            echo "❌ XGBoost 빠른 테스트 실패"
            exit 1
        fi
        
        echo ""
        echo ">>> CatBoost 빠른 테스트"
        python scripts/run_hyperparameter_tuning.py \
            --model-type catboost \
            --experiment-type hyperparameter_tuning \
            --nrows 1000 \
            --n-trials 10 \
            --experiment-name "quick_test_catboost" \
            --verbose 2
        
        if [ $? -ne 0 ]; then
            echo "❌ CatBoost 빠른 테스트 실패"
            exit 1
        fi
        
        echo ""
        echo ">>> 리샘플링 빠른 테스트"
        python scripts/run_resampling_experiment.py \
            --model-type lightgbm \
            --resampling-method smote \
            --nrows 1000 \
            --n-trials 10 \
            --experiment-name "quick_test_resampling_lightgbm" \
            --verbose 2
        
        if [ $? -ne 0 ]; then
            echo "❌ 리샘플링 빠른 테스트 실패"
            exit 1
        fi
        
        echo ""
        echo "✅ 빠른 테스트 완료! MLflow UI에서 결과를 확인하세요."
        ;;
        
    4)
        echo "🔧 환경 설정 및 초기 분석"
        echo ""
        echo ">>> 시스템 상태 확인"
        python scripts/check_cpu_usage.py
        
        echo ""
        echo ">>> 데이터 분석 실행"
        if python src/data_analysis.py; then
            echo "✅ 데이터 분석 완료!"
            echo "결과는 data/sourcedata_analysis/ 폴더에 저장되었습니다."
        else
            echo "❌ 데이터 분석 중 오류 발생"
        fi
        
        echo ""
        echo ">>> MLflow 실험 상태 확인"
        python scripts/cleanup_mlflow_experiments.py --action list
        ;;
        
    5)
        echo "📊 실험 상태 확인 및 MLflow UI"
        echo ""
        echo ">>> 현재 MLflow 실험 목록"
        python scripts/cleanup_mlflow_experiments.py --action list
        
        echo ""
        echo ">>> MLflow UI 실행"
        echo "MLflow UI가 http://localhost:5000 에서 실행됩니다."
        mlflow ui --host 0.0.0.0 --port 5000
        ;;
        
    6)
        echo "🧹 실험 중단 및 정리"
        echo ""
        echo "현재 실행 중인 실험 프로세스를 정리합니다..."
        pkill -f "run_hyperparameter_tuning"
        pkill -f "run_resampling_experiment"
        echo "✅ 프로세스 정리 완료"
        
        echo ""
        echo "MLflow 실험 정리를 원하시나요?"
        read -p "오래된 실험을 정리하시겠습니까? (y/n): " -n 1 -r cleanup_mlflow
        echo
        if [[ $cleanup_mlflow =~ ^[Yy]$ ]]; then
            echo ">>> MLflow 실험 정리"
            python scripts/cleanup_mlflow_experiments.py --action cleanup --days-old 7 --force
        fi
        ;;
        
    *)
        echo "잘못된 선택입니다."
        exit 1
        ;;
esac

echo ""
echo "=================================================================="
echo "실험 스크립트 실행이 완료되었습니다."
echo ""
echo "📁 결과 파일들:"
echo "- MLflow 실험: mlruns/ 폴더"
echo "- 시각화: results/visualizations/ 폴더"
echo "- 학습된 모델: models/ 폴더"
echo "- 실험 로그: logs/ 폴더"
echo ""
echo "🔗 유용한 명령어:"
echo "- MLflow UI: mlflow ui --host 0.0.0.0 --port 5000"
echo "- 실험 정리: python scripts/cleanup_mlflow_experiments.py --action list"
echo "- CPU 확인: python scripts/check_cpu_usage.py"
echo ""
echo "📋 다음 단계:"
echo "1. MLflow UI에서 실험 결과 분석"
echo "2. 최고 성능 모델 선정"
echo "3. 앙상블 모델 구축 (필요시)"
echo "4. 모델 배포 준비"
echo "=================================================================="

# 시그널 핸들러 제거
trap - INT TERM