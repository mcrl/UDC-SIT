# SIT4-DISCNet train 방법

**중점적으로 수정 / 확인하셔야 하는 파일들은 `ky`로 시작합니다.**

## 시작 전 준비사항

해당 섹션의 파일경로는 `/home/chanwoo/modified-DISCNet` 기준입니다.

```bash
cd /home/chanwoo/modified-DISCNet
```

### 데이터셋 목록 준비

+ train / validation에 사용할 파일명 목록을 정리한 파일
    + `train`을 위해서 1개, `validation`을 위해 1개.
    + `path/to/train/info/file` 내용물 예시

        ```plain
        1.npy
        2.npy`
        4.npy
        5.npy
        ````

        혹은

        ```plain
        1.npy /home/chanwoo/sit-train-psf.npy
        2.npy /home/chanwoo/sit-train-psf.npy
        4.npy /home/chanwoo/sit-train-psf.npy
        5.npy /home/chanwoo/sit-train-psf.npy
        ```
    + 훈련 스크립트는 위의 형태를 아래의 형태로 자동으로 변환해 줄 것입니다.

### 옵션 파일 수정

+ `option/train/ky-train-options.yml` 작성
    + `kyusu` 가 적혀있는 행 / 섹션을 중심으로 보시면 됩니다. (`ctrl+f`) 사용
    + `option/train/ky-DISCNet-DSC-template.yml`에 템플릿이 있습니다.
    + 데이터 디렉토리, 저장할 validation 이미지의 비율 등을 조절할 수 있습니다.
    + 옵션 파일명은 다르게 지정할 수 있습니다.


### 훈련 스크립트 수정

+ `ky-train-template.sh`를 참조하여 `ky-train-script.sh`를 작성합니다.
    ```bash
    # ...other commands...

    # kyusu: Change option path as you want
    option=options/train/ky-train-options.yml

    # ...other commands...
    ```

## 훈련 명령어

```bash
cd /home/chanwoo/modified-DISCNet

# 화면에 나오는 모든 내용을 로그로 남겨 줍니다.
script -a your-logfile-name.log

# [경고] 이 명령어를 수행한 후에는 아래의 명령어를 절대 수행하지 마세요!
# cat your-logfile-name.log

# 반드시 script 명령어를 수행한 후 시작
conda activate DISCNet

# 훈련 명령어
bash ky-train-script.sh # 훈련이 끝날 때까지 자동으로 훈련을 재시작해 줄 것입니다.
```

## 로그 분석

> 해당 단계를 수행해야 csv 파일을 얻을 수 있습니다.

`your-logfile-name.log`를 준비합니다.

```bash
cd /home/chanwoo
python parse-log.py --log modified-DISCNet/your-logfile-name.log --dest path/to/csv/file.csv
```

