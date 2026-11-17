// Decode off-screen so the displayed image stays intact until its replacement is ready.
class FramePlayer {
    constructor(element) {
        this.element = element;
        this.generation = 0;
        this.sequence = 0;
        this.displayedSequence = 0;
        this.displayedUrl = null;
        this.pending = new Map();
    }

    reset() {
        this.generation++;
        this.element.removeAttribute('src');
        if (this.displayedUrl) URL.revokeObjectURL(this.displayedUrl);
        this.displayedUrl = null;
        for (const finish of [...this.pending.values()]) finish();
    }

    async receive(bytes, acknowledge) {
        if (document.hidden) {
            if (typeof acknowledge === 'function') acknowledge();
            return;
        }
        const generation = this.generation;
        const sequence = ++this.sequence;
        const url = URL.createObjectURL(new Blob([bytes], { type: 'image/jpeg' }));
        let finished = false;
        let displayed = false;
        const finish = () => {
            if (finished) return;
            finished = true;
            this.pending.delete(sequence);
            if (!displayed) URL.revokeObjectURL(url);
            if (typeof acknowledge === 'function') acknowledge();
        };
        this.pending.set(sequence, finish);
        try {
            const image = new Image();
            image.src = url;
            await image.decode();
            if (generation !== this.generation || document.hidden || sequence < this.displayedSequence) return;
            const previous = this.displayedUrl;
            this.element.src = url;
            this.displayedUrl = url;
            this.displayedSequence = sequence;
            displayed = true;
            if (previous) URL.revokeObjectURL(previous);
        } catch {
            // Keep the last good frame and release credit even when decoding fails.
        } finally {
            finish();
        }
    }
}
