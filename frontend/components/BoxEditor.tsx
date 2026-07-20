"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

export interface EditorBox {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
}

interface BoxEditorProps {
  imageUrl: string;
  naturalWidth: number;
  naturalHeight: number;
  boxes: EditorBox[];
  onChange: (boxes: EditorBox[]) => void;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}

type Corner = "nw" | "ne" | "sw" | "se";

interface DragState {
  mode: "move" | "resize" | "draw";
  startX: number; // natural coords
  startY: number;
  orig?: EditorBox;
  corner?: Corner;
  drawId?: string;
}

const MIN_SIZE = 12; // natural px

let _idCounter = 0;
const newId = () => `box_${Date.now()}_${_idCounter++}`;

export default function BoxEditor({
  imageUrl,
  naturalWidth,
  naturalHeight,
  boxes,
  onChange,
  selectedId,
  onSelect,
}: BoxEditorProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);
  const dragRef = useRef<DragState | null>(null);
  // keep latest boxes in a ref so global listeners see fresh data
  const boxesRef = useRef(boxes);
  boxesRef.current = boxes;

  const measure = useCallback(() => {
    if (wrapRef.current) {
      setScale(wrapRef.current.clientWidth / naturalWidth);
    }
  }, [naturalWidth]);

  useLayoutEffect(() => {
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [measure]);

  const toNatural = (e: PointerEvent | React.PointerEvent) => {
    const rect = wrapRef.current!.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) / scale,
      y: (e.clientY - rect.top) / scale,
    };
  };

  const clamp = (b: EditorBox): EditorBox => {
    const x = Math.max(0, Math.min(b.x, naturalWidth - MIN_SIZE));
    const y = Math.max(0, Math.min(b.y, naturalHeight - MIN_SIZE));
    const w = Math.max(MIN_SIZE, Math.min(b.w, naturalWidth - x));
    const h = Math.max(MIN_SIZE, Math.min(b.h, naturalHeight - y));
    return { ...b, x, y, w, h };
  };

  const onPointerMove = useCallback(
    (e: PointerEvent) => {
      const drag = dragRef.current;
      if (!drag) return;
      const { x, y } = toNatural(e);
      const dx = x - drag.startX;
      const dy = y - drag.startY;

      if (drag.mode === "move" && drag.orig) {
        const moved = clamp({ ...drag.orig, x: drag.orig.x + dx, y: drag.orig.y + dy });
        onChange(boxesRef.current.map((b) => (b.id === drag.orig!.id ? moved : b)));
      } else if (drag.mode === "resize" && drag.orig && drag.corner) {
        let { x: bx, y: by, w: bw, h: bh } = drag.orig;
        if (drag.corner.includes("e")) bw = drag.orig.w + dx;
        if (drag.corner.includes("s")) bh = drag.orig.h + dy;
        if (drag.corner.includes("w")) { bx = drag.orig.x + dx; bw = drag.orig.w - dx; }
        if (drag.corner.includes("n")) { by = drag.orig.y + dy; bh = drag.orig.h - dy; }
        const resized = clamp({ ...drag.orig, x: bx, y: by, w: bw, h: bh });
        onChange(boxesRef.current.map((b) => (b.id === drag.orig!.id ? resized : b)));
      } else if (drag.mode === "draw" && drag.drawId) {
        const nx = Math.min(drag.startX, x);
        const ny = Math.min(drag.startY, y);
        const nw = Math.abs(x - drag.startX);
        const nh = Math.abs(y - drag.startY);
        const drawn = clamp({ id: drag.drawId, x: nx, y: ny, w: nw, h: nh });
        onChange(boxesRef.current.map((b) => (b.id === drag.drawId ? drawn : b)));
      }
    },
    [scale, onChange, naturalWidth, naturalHeight] // eslint-disable-line react-hooks/exhaustive-deps
  );

  const onPointerUp = useCallback(() => {
    const drag = dragRef.current;
    if (drag?.mode === "draw" && drag.drawId) {
      // discard boxes that are too small (a click, not a drag)
      const drawn = boxesRef.current.find((b) => b.id === drag.drawId);
      if (drawn && (drawn.w < MIN_SIZE * 2 || drawn.h < MIN_SIZE * 2)) {
        onChange(boxesRef.current.filter((b) => b.id !== drag.drawId));
        onSelect(null);
      }
    }
    dragRef.current = null;
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", onPointerUp);
  }, [onPointerMove, onChange, onSelect]);

  const beginDrag = (drag: DragState) => {
    dragRef.current = drag;
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
  };

  const onBackgroundDown = (e: React.PointerEvent) => {
    if (e.target !== e.currentTarget) return; // clicked a box/handle
    const { x, y } = toNatural(e);
    const id = newId();
    onChange([...boxesRef.current, { id, x, y, w: 0, h: 0 }]);
    onSelect(id);
    beginDrag({ mode: "draw", startX: x, startY: y, drawId: id });
  };

  const onBoxDown = (e: React.PointerEvent, box: EditorBox) => {
    e.stopPropagation();
    onSelect(box.id);
    const { x, y } = toNatural(e);
    beginDrag({ mode: "move", startX: x, startY: y, orig: box });
  };

  const onHandleDown = (e: React.PointerEvent, box: EditorBox, corner: Corner) => {
    e.stopPropagation();
    onSelect(box.id);
    const { x, y } = toNatural(e);
    beginDrag({ mode: "resize", startX: x, startY: y, orig: box, corner });
  };

  // delete selected box with Delete/Backspace
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.key === "Delete" || e.key === "Backspace") && selectedId) {
        onChange(boxesRef.current.filter((b) => b.id !== selectedId));
        onSelect(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedId, onChange, onSelect]);

  const corners: Corner[] = ["nw", "ne", "sw", "se"];
  const cornerPos: Record<Corner, string> = {
    nw: "-top-1.5 -left-1.5 cursor-nwse-resize",
    ne: "-top-1.5 -right-1.5 cursor-nesw-resize",
    sw: "-bottom-1.5 -left-1.5 cursor-nesw-resize",
    se: "-bottom-1.5 -right-1.5 cursor-nwse-resize",
  };

  return (
    <div
      ref={wrapRef}
      className="relative w-full select-none touch-none"
      style={{ aspectRatio: `${naturalWidth} / ${naturalHeight}` }}
      onPointerDown={onBackgroundDown}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={imageUrl}
        alt="page"
        className="absolute inset-0 w-full h-full pointer-events-none"
        onLoad={measure}
        draggable={false}
      />
      {boxes.map((b, i) => {
        const sel = b.id === selectedId;
        return (
          <div
            key={b.id}
            onPointerDown={(e) => onBoxDown(e, b)}
            className={`absolute box-border cursor-move ${
              sel ? "border-2 border-blue-400 bg-blue-400/10 z-20" : "border-2 border-emerald-400/80 bg-emerald-400/5 z-10 hover:border-emerald-300"
            }`}
            style={{
              left: b.x * scale,
              top: b.y * scale,
              width: b.w * scale,
              height: b.h * scale,
            }}
          >
            <span className="absolute -top-5 left-0 text-[11px] font-mono px-1 rounded bg-slate-900/90 text-emerald-300">
              {i + 1}
            </span>
            {sel &&
              corners.map((c) => (
                <span
                  key={c}
                  onPointerDown={(e) => onHandleDown(e, b, c)}
                  className={`absolute w-3 h-3 rounded-sm bg-blue-400 border border-white ${cornerPos[c]}`}
                />
              ))}
          </div>
        );
      })}
    </div>
  );
}
