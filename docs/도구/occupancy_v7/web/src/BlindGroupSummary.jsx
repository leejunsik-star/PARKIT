import React from 'react';

export default function BlindGroupSummary({groups}) {
  if (!groups?.length) return null;
  return <ul className="blind-group-summary" aria-label="사각지대 점유 추정">
    {groups.map(group => <li key={group.zone_id}>
      <strong>{group.zone_name} 사각지대 · {group.occupied}/{group.capacity}면 점유 추정</strong>
      <span>{group.slots.join(', ')}번 중 어느 칸에 주차했는지 확인할 수 없어요.</span>
    </li>)}
  </ul>;
}
