"""Read a PARKIT checkpoint without changing it; export a readable summary."""
import argparse
import hashlib
from pathlib import Path

import torch

from .model import ParkitTiny


def summarize(source, output=None):
    source = Path(source).resolve()
    output = Path(output).resolve() if output else source.with_suffix('.summary.md')
    if source == output:
        raise ValueError('요약 파일 경로는 원본 가중치 경로와 달라야 합니다.')
    if output.suffix.lower() != '.md':
        raise ValueError('요약 출력 파일은 .md 확장자를 사용하세요.')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or not {'model', 'width', 'size'} <= checkpoint.keys():
        raise ValueError('지원하는 PARKIT Tiny 체크포인트 형식이 아닙니다.')
    model = ParkitTiny(checkpoint['width'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    state = model.state_dict()
    for name, value in state.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f'유한하지 않은 값이 있습니다: {name}')
    parameters = sum(p.numel() for p in model.parameters())
    lines = [
        '# PARKIT Tiny 가중치 확인 결과', '',
        f'- 원본 파일: `{source.name}`',
        '- 실제 모델 형식: **PARKIT Tiny** (Ultralytics YOLO 형식이 아님)',
        '- 검사: 안전한 가중치 로딩, 모델 구조와 strict 일치, 유한한 텐서 값 확인 통과',
        f'- 파일 크기: {source.stat().st_size:,} bytes',
        f'- 파라미터 수: {parameters:,}',
        f'- 모델 폭: {checkpoint["width"]}',
        f'- 입력 크기: {checkpoint["size"]} × {checkpoint["size"]}',
        f'- 저장된 학습 회차: {checkpoint.get("epoch", "기록 없음")}',
        f'- seed: {checkpoint.get("seed", "기록 없음")}',
        f'- 과적합 검사 전용: {checkpoint.get("overfit_only", "기록 없음")}',
        f'- SHA-256: `{digest}`', '',
        '이 파일은 가중치의 읽기 전용 요약입니다. 요약을 편집해도 모델은 바뀌지 않습니다.',
        '원본 .pt는 텍스트가 아니라 학습된 숫자 배열과 메타데이터를 담는 바이너리입니다.',
        '로딩 검사는 파일이 현재 모델 코드와 호환됨을 확인하며 탐지 정확도를 보장하지는 않습니다.',
        '모델 구조를 수정하려면 parkit_detector/model.py를 편집하고 다시 학습해야 합니다.', '',
        '## 신경망 구조', '', '```text', str(model), '```', '',
        '## 저장된 텐서 목록', '',
        '학습 파라미터 외에 BatchNorm 통계와 카운터 등 버퍼도 포함합니다.', '',
        '| 이름 | 모양 | 자료형 | 원소 수 |',
        '|---|---|---|---:|',
    ]
    for name, value in state.items():
        lines.append(f'| `{name}` | `{list(value.shape)}` | `{value.dtype}` | {value.numel():,} |')
    # Guard against accidental source modification while inspecting.
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise RuntimeError('검사 도중 원본 파일이 변경되어 요약을 저장하지 않았습니다.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'검사 통과: {source.name} → {output}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('weights', help='확인할 PARKIT .pt 파일')
    parser.add_argument('--out', help='출력 .md 경로 (기본: 원본 이름.summary.md)')
    args = parser.parse_args()
    torch.set_num_threads(2)
    summarize(args.weights, args.out)


if __name__ == '__main__':
    main()
